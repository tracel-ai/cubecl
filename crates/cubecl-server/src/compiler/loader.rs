//! The kernels a device has loaded, and the one path that loads more.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use cubecl_common::profile::Instant;
use cubecl_environment::collections::{HashMap, HashSet};

use super::{
    ArtifactCompiler, ArtifactId, BatchOutcome, CompilationBatchRecording, CompilationOutcome,
    CompilationRecording, CompilationTarget, VariantOf,
};
use crate::config::RuntimeConfig;
use crate::kernel::CubeKernel;
use crate::logging::ServerLogger;
use crate::server::LaunchError;

/// Every kernel a device has loaded, in front of the backend that compiles and
/// loads the ones it has not.
///
/// A kernel it does not hold is read from the compilation store, or failing
/// that compiled and stored, then loaded — in that order on every backend, and
/// recorded the same way: what differs between backends is the
/// [`CompilationTarget`] it is generic over.
///
/// Kernels can also be [queued](Self::enqueue) and compiled together: the
/// half of the work any thread can do then runs on as many threads as
/// there are kernels, up to the configured
/// [parallelism](crate::config::compilation::CompilationConfig::parallelism),
/// each taking the next kernel as it finishes one. However long the queue,
/// it compiles in one phase that lasts about as long as its slowest kernel.
pub struct KernelLoader<T: CompilationTarget> {
    target: T,
    loaded: CompilationCache<ArtifactId<VariantOf<T>>, T::Loaded>,
    /// The kernels queued and not compiled yet.
    queue: KernelQueue<VariantOf<T>>,
    /// Why each queued kernel of the latest batch that failed to compile
    /// failed, until its own launch reports it: the launch is where the
    /// failure has outputs to carry, and compiling the kernel again there
    /// would only fail again. The next batch drops what is left, so a kernel
    /// whose launch never came compiles again rather than keeping a stale
    /// error: what the launch is spared is one compile, not every later one.
    failed: HashMap<ArtifactId<VariantOf<T>>, LaunchError>,
    /// How many threads compile at once.
    parallelism: usize,
}

impl<T: CompilationTarget> KernelLoader<T> {
    /// A loader holding nothing yet, bound to the active environment exactly
    /// when `target` [persists](CompilationTarget::persists) what it compiles.
    pub fn new(target: T) -> Self {
        let loaded = CompilationCache::new(target.persists());
        let parallelism = crate::config::CubeClRuntimeConfig::get()
            .compilation
            .parallelism();

        Self {
            target,
            loaded,
            queue: KernelQueue::new(),
            failed: HashMap::new(),
            parallelism,
        }
    }

    /// The kernel loaded for `id`, if it is.
    #[cfg(test)]
    fn get(&mut self, id: &ArtifactId<VariantOf<T>>) -> Option<&T::Loaded> {
        self.loaded.get(id)
    }

    /// The kernel loaded for `kernel`, under `id`, loading it first if it is
    /// not. A kernel is about to execute, so every queued kernel is compiled
    /// first — together with this one when it needs compiling.
    ///
    /// `id` is `kernel`'s, with what the launch adds to it: the launch already
    /// built the kernel's id, and building it again on every launch costs. A
    /// kernel already loaded, with nothing queued, costs one lookup and a
    /// clone of what was loaded; `id` is only cloned to compile.
    ///
    /// # Errors
    ///
    /// The validation, compilation or load of `kernel` that failed, now or
    /// when it was queued. The failure is reported once and not kept, so the
    /// next launch of the kernel tries again.
    pub fn load(
        &mut self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<VariantOf<T>>,
        logger: &ServerLogger,
    ) -> Result<T::Loaded, LaunchError> {
        if self.queue.is_empty()
            && let Some(loaded) = self.loaded.get(id)
        {
            return Ok(loaded.clone());
        }
        self.load_missing(kernel, id, logger)
    }

    /// [`load`](Self::load) past its one lookup: the kernel is not loaded, or
    /// the queue has to compile first.
    #[cold]
    fn load_missing(
        &mut self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<VariantOf<T>>,
        logger: &ServerLogger,
    ) -> Result<T::Loaded, LaunchError> {
        loop {
            // A failure is reported as it stands only when nothing else is
            // queued: a batch drops it, and the kernel compiles in that batch.
            let reported = self.failed.contains_key(id) && self.queue.is_empty();
            let missing = !self.loaded.contains(id) && !reported;
            let asked = missing.then(|| Request {
                id: id.clone(),
                kernel,
            });
            self.compile_queue(asked, logger);

            if let Some(err) = self.failed.remove(id) {
                return Err(err);
            }
            // An environment switch during the batch drops every kernel loaded
            // before it, this one too when it was: it loads again, from the
            // store of the environment switched to.
            if let Some(loaded) = self.loaded.get(id) {
                return Ok(loaded.clone());
            }
        }
    }

    /// Compiles every queued kernel now, rather than inside the next
    /// [`load`](Self::load), so the launch that follows pays for its own
    /// kernel only. A queued kernel that fails reports it when it is launched.
    pub fn compile_queued(&mut self, logger: &ServerLogger) {
        self.compile_queue(None, logger);
    }

    /// Compiles the queue and `asked` together, keeping each kernel or why it
    /// failed.
    fn compile_queue(&mut self, asked: Option<Request<'_, VariantOf<T>>>, logger: &ServerLogger) {
        if self.queue.is_empty() && asked.is_none() {
            return;
        }
        let queue = self.queue.take();
        // A failure an earlier batch left unreported is dropped: its kernel
        // compiles again when it is launched or queued.
        self.failed.clear();
        let mut requests: Vec<Request<'_, VariantOf<T>>> = queue
            .iter()
            .filter(|queued| asked.as_ref().is_none_or(|asked| queued.id != asked.id))
            .map(Queued::request)
            .collect();
        requests.extend(asked);
        for outcome in self.compile(requests, logger) {
            self.keep(outcome);
        }
    }

    /// Queues `kernel` under `variant`, to be compiled with the others when the
    /// next kernel is [loaded](Self::load) for a launch, and only then. One
    /// already loaded or queued is left alone; one that failed is tried again,
    /// its failure dropped.
    ///
    /// The queue does not compile on its own when it grows: one phase over
    /// everything queued lasts about as long as its slowest kernel, where
    /// two phases would each wait on theirs.
    pub fn enqueue(&mut self, kernel: Box<dyn CubeKernel>, variant: VariantOf<T>) {
        let id = ArtifactId {
            kernel: kernel.id(),
            variant,
        };
        if self.loaded.contains(&id) || self.queue.contains(&id) {
            return;
        }
        self.failed.remove(&id);
        self.queue.push(id, kernel);
    }

    /// Keeps a loaded kernel, or the reason it failed.
    fn keep(&mut self, outcome: JobOutcome<T>) {
        match outcome.result {
            Ok(loaded) => self.loaded.insert(outcome.id, loaded),
            Err(err) => {
                self.failed.insert(outcome.id, err);
            }
        }
    }

    /// Obtains each requested kernel's artifact from the store or the
    /// compiler, and loads it.
    ///
    /// The store and the loads are reached in order, on this thread; lowering
    /// and finalizing run on several threads at once when more than one
    /// kernel has that step to take.
    fn compile(
        &mut self,
        requests: Vec<Request<'_, VariantOf<T>>>,
        logger: &ServerLogger,
    ) -> Vec<JobOutcome<T>> {
        let batch = CompilationBatchRecording::new();
        let mut jobs: Vec<Job<'_, T>> = requests.into_iter().map(Job::new).collect();
        let jobs_len = jobs.len();

        for job in jobs.iter_mut() {
            job.look_up(&mut self.target);
        }

        let compiler = self.target.compiler();
        let mut missing: Vec<&mut Job<'_, T>> =
            jobs.iter_mut().filter(|job| job.is_missing()).collect();
        let threads = self.parallelism.min(missing.len()).max(1);
        if missing.len() > 1 {
            log::info!(
                target: "cubecl::compilation",
                "Compiling {} kernels together ({} asked, the rest read from the store) on {} threads",
                missing.len(),
                jobs_len,
                threads,
            );
        }
        let reuses = self.target.persists();
        for_each_at_once(&mut missing, self.parallelism, |job| {
            job.lower(compiler, logger);
            // With no artifact to reuse it by, a kernel finalizes on the thread
            // that lowered it, without waiting for the rest of the batch to
            // lower: the batch lasts about as long as its slowest kernel, not
            // its slowest lowering plus its slowest finalizing.
            if job.is_lowered() && !(reuses && job.has_source()) {
                job.finalize(compiler);
            }
        });

        let mut finalizing = HashSet::new();
        for job in jobs.iter_mut() {
            job.reuse(&mut self.target, &mut finalizing);
        }

        let leading: Vec<(usize, String)> = jobs
            .iter()
            .enumerate()
            .filter_map(|(index, job)| match &job.step {
                Step::Lowered {
                    source: Some(source),
                    ..
                } => Some((index, source.clone())),
                _ => None,
            })
            .collect();
        let compiler = self.target.compiler();
        let mut lowered: Vec<&mut Job<'_, T>> =
            jobs.iter_mut().filter(|job| job.is_lowered()).collect();
        for_each_at_once(&mut lowered, self.parallelism, |job| job.finalize(compiler));

        // A kernel waiting on a source that failed to finalize fails the same
        // way, rather than finalizing it again, one after another, on this
        // thread.
        let failed: HashMap<String, LaunchError> = leading
            .into_iter()
            .filter_map(|(index, source)| match &jobs[index].step {
                Step::Failed(err) => Some((source, err.clone())),
                _ => None,
            })
            .collect();
        for job in jobs.iter_mut() {
            if let Step::Waiting { source, .. } = &job.step
                && let Some(err) = failed.get(source)
            {
                job.step = Step::Failed(err.clone());
            }
        }

        let outcomes: Vec<JobOutcome<T>> = jobs
            .into_iter()
            .map(|job| job.load(&mut self.target))
            .collect();
        batch.close(BatchOutcome {
            kernels: jobs_len,
            threads,
            stored: outcomes.iter().any(|outcome| outcome.stored),
        });
        outcomes
    }
}

impl<T: CompilationTarget + core::fmt::Debug> core::fmt::Debug for KernelLoader<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The loaded kernels are driver handles, which say nothing in a log.
        f.debug_struct("KernelLoader")
            .field("target", &self.target)
            .field("queued", &self.queue.len())
            .field("failed", &self.failed.len())
            .field("parallelism", &self.parallelism)
            .finish_non_exhaustive()
    }
}

/// How a [`Job`] ended: its kernel loaded, or why it was not.
struct JobOutcome<T: CompilationTarget> {
    id: ArtifactId<VariantOf<T>>,
    result: Result<T::Loaded, LaunchError>,
    /// Whether the compilation store took its artifact.
    stored: bool,
}

/// The kernels a [`KernelLoader`] holds loaded — pipelines, modules — in front
/// of the compilation store its target persists to.
///
/// Entries are dropped when the environment switches, because the map is bound
/// to an environment exactly as the store it mirrors is. One served after a
/// switch would describe the environment that is gone, and, worse, would never
/// be written to the new environment's store, so a bundle exported from that
/// environment would silently be missing that kernel. This is the same contract
/// a [`Store`](cubecl_environment::persistence::Store) applies to itself, for
/// the state a store cannot see — see
/// [`cubecl_environment::environment::generation`].
///
/// Every accessor resets before it answers, so the loader has nothing to
/// remember beyond using this in place of a plain map.
#[derive(Debug)]
struct CompilationCache<K, V> {
    entries: HashMap<K, V>,
    /// The generation the entries were built under, or `None` when the cache
    /// mirrors no store and so is unbound.
    generation: Option<u32>,
}

impl<K: Eq + core::hash::Hash, V> CompilationCache<K, V> {
    /// An empty cache, bound to the active environment when `bound`: when
    /// its artifacts persist in a store that a switch replaces. One with no
    /// persistent store to mirror is never reset — with nothing persisted, a
    /// switch changes nothing about what it holds, so resetting it would only
    /// buy a redundant compilation, the same reason the autotune cache
    /// survives a switch when its persistent cache is off.
    fn new(bound: bool) -> Self {
        Self {
            entries: HashMap::new(),
            generation: bound.then(cubecl_environment::environment::generation),
        }
    }

    /// The kernel loaded for `key`, if it is still valid.
    fn get(&mut self, key: &K) -> Option<&V> {
        self.reset_if_switched();
        self.entries.get(key)
    }

    /// Whether a kernel is loaded for `key` and still valid.
    fn contains(&mut self, key: &K) -> bool {
        self.reset_if_switched();
        self.entries.contains_key(key)
    }

    /// Keeps a freshly loaded kernel.
    fn insert(&mut self, key: K, value: V) {
        self.reset_if_switched();
        self.entries.insert(key, value);
    }

    /// Drops every entry when the environment switched since the last access,
    /// adopting the new generation so one switch costs one reset.
    fn reset_if_switched(&mut self) {
        let Some(generation) = self.generation else {
            return;
        };

        let current = cubecl_environment::environment::generation();
        if current == generation {
            return;
        }

        log::debug!("Environment switched, dropping the in-memory compilation cache");
        self.generation = Some(current);
        self.entries.clear();
    }
}

/// The kernels queued for compilation and not compiled yet, each once.
struct KernelQueue<V> {
    kernels: Vec<Queued<V>>,
    /// Their ids, so queuing one more costs a lookup rather than a scan.
    ids: HashSet<ArtifactId<V>>,
}

impl<V: Clone + Eq + core::hash::Hash> KernelQueue<V> {
    fn new() -> Self {
        Self {
            kernels: Vec::new(),
            ids: HashSet::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    fn len(&self) -> usize {
        self.kernels.len()
    }

    fn contains(&self, id: &ArtifactId<V>) -> bool {
        self.ids.contains(id)
    }

    /// Queues `kernel` under `id`, which is not queued yet.
    fn push(&mut self, id: ArtifactId<V>, kernel: Box<dyn CubeKernel>) {
        self.ids.insert(id.clone());
        self.kernels.push(Queued { id, kernel });
    }

    /// Every queued kernel, leaving the queue empty.
    fn take(&mut self) -> Vec<Queued<V>> {
        self.ids.clear();
        core::mem::take(&mut self.kernels)
    }
}

/// A kernel queued for compilation.
struct Queued<V> {
    id: ArtifactId<V>,
    kernel: Box<dyn CubeKernel>,
}

impl<V: Clone> Queued<V> {
    fn request(&self) -> Request<'_, V> {
        Request {
            id: self.id.clone(),
            kernel: &*self.kernel,
        }
    }
}

/// A kernel to compile, queued or asked for.
struct Request<'k, V> {
    id: ArtifactId<V>,
    kernel: &'k dyn CubeKernel,
}

/// One kernel on its way through [`KernelLoader::compile`].
struct Job<'k, T: CompilationTarget> {
    id: ArtifactId<VariantOf<T>>,
    kernel: &'k dyn CubeKernel,
    recording: CompilationRecording,
    step: Step<T>,
}

/// How far a [`Job`] got.
enum Step<T: CompilationTarget> {
    /// The store holds nothing for it: it has to be compiled.
    Missing,
    /// Lowered, with the text its artifact can be reused by.
    Lowered {
        lowered: <T::Compiler as ArtifactCompiler>::Lowered,
        source: Option<String>,
    },
    /// Lowered to the same source as an earlier kernel of the batch, whose
    /// artifact it reuses once that one is stored rather than finalizing the
    /// same text twice.
    Waiting {
        lowered: <T::Compiler as ArtifactCompiler>::Lowered,
        source: String,
    },
    /// An artifact to load, and how it was obtained.
    Ready {
        artifact: <T::Compiler as ArtifactCompiler>::Artifact,
        source: Option<String>,
        outcome: CompilationOutcome,
    },
    Failed(LaunchError),
}

impl<'k, T: CompilationTarget> Job<'k, T> {
    fn new(request: Request<'k, VariantOf<T>>) -> Self {
        Self {
            recording: CompilationRecording::new(&request.id.kernel),
            id: request.id,
            kernel: request.kernel,
            step: Step::Missing,
        }
    }

    fn is_missing(&self) -> bool {
        matches!(self.step, Step::Missing)
    }

    fn is_lowered(&self) -> bool {
        matches!(self.step, Step::Lowered { .. })
    }

    /// Whether it lowered to text an artifact can be reused by.
    fn has_source(&self) -> bool {
        matches!(
            self.step,
            Step::Lowered {
                source: Some(_),
                ..
            }
        )
    }

    /// Takes what the store holds for the kernel, if it holds anything.
    fn look_up(&mut self, target: &mut T) {
        if let Some(artifact) = target.stored(&self.id) {
            self.step = Step::Ready {
                artifact,
                source: None,
                outcome: CompilationOutcome::Loaded,
            };
        }
    }

    /// Lowers a kernel the store did not hold.
    fn lower(&mut self, compiler: &T::Compiler, logger: &ServerLogger) {
        let started = Instant::now();
        self.step = match compiler.lower(self.kernel, &self.id, &mut self.recording, logger) {
            Ok(lowered) => Step::Lowered {
                source: T::Compiler::source(&lowered).map(String::from),
                lowered,
            },
            Err(err) => Step::Failed(err),
        };
        self.recording.worked(started.elapsed());
    }

    /// Takes an artifact another kernel finalized from the same source, if
    /// the store holds one; or, when an earlier kernel of this batch already
    /// took that source or is about to finalize it, waits to take it from the
    /// store once that kernel is stored. `finalizing` is the sources the batch
    /// takes or finalizes so far. Without a store there is nothing to take or
    /// wait for, and every kernel finalizes its own.
    fn reuse(&mut self, target: &mut T, finalizing: &mut HashSet<String>) {
        let Step::Lowered {
            source: Some(source),
            ..
        } = &self.step
        else {
            return;
        };
        if !target.persists() {
            return;
        }

        if finalizing.contains(source) {
            let Step::Lowered {
                lowered,
                source: Some(source),
            } = core::mem::replace(&mut self.step, Step::Missing)
            else {
                unreachable!("matched right above");
            };
            self.step = Step::Waiting { lowered, source };
            return;
        }

        finalizing.insert(source.clone());
        if let Some(artifact) = target.stored_for_source(source) {
            self.step = Step::Ready {
                artifact,
                source: Some(source.clone()),
                outcome: CompilationOutcome::Rekeyed,
            };
        }
    }

    /// Finalizes a lowered kernel nothing could be reused for.
    fn finalize(&mut self, compiler: &T::Compiler) {
        let Step::Lowered { lowered, source } = core::mem::replace(&mut self.step, Step::Missing)
        else {
            unreachable!("only lowered jobs are finalized");
        };

        let started = Instant::now();
        self.step = match compiler.finalize(&self.id, lowered) {
            Ok(artifact) => Step::Ready {
                artifact,
                source,
                outcome: CompilationOutcome::Compiled,
            },
            Err(err) => Step::Failed(err),
        };
        self.recording.worked(started.elapsed());
    }

    /// Loads the artifact on the device, keeps it in the store when it was
    /// not read from there, and closes the record.
    ///
    /// A waiting job takes what the kernel it waited for stored, now that the
    /// batch has stored it, and finalizes its own only when there is no store
    /// to take it from.
    fn load(mut self, target: &mut T) -> JobOutcome<T> {
        if let Step::Waiting { .. } = self.step {
            let Step::Waiting { lowered, source } =
                core::mem::replace(&mut self.step, Step::Missing)
            else {
                unreachable!("matched right above");
            };
            self.step = match target.stored_for_source(&source) {
                Some(artifact) => Step::Ready {
                    artifact,
                    source: Some(source),
                    outcome: CompilationOutcome::Rekeyed,
                },
                None => Step::Lowered {
                    lowered,
                    source: Some(source),
                },
            };
            if self.is_lowered() {
                self.finalize(target.compiler());
            }
        }

        let mut stored = false;
        let result = match self.step {
            Step::Ready {
                artifact,
                source,
                outcome,
            } => {
                let started = Instant::now();
                match target.load(&self.id, &artifact) {
                    Ok(loaded) => {
                        self.recording.worked(started.elapsed());
                        // What the store gave is in it already.
                        if outcome != CompilationOutcome::Loaded {
                            stored = target.store(&self.id, artifact, source.as_deref());
                        }
                        self.recording.close(outcome, stored);
                        Ok(loaded)
                    }
                    Err(err) => Err(err.into()),
                }
            }
            Step::Failed(err) => Err(err),
            Step::Missing | Step::Lowered { .. } | Step::Waiting { .. } => {
                unreachable!("every job is lowered, then finalized or reused, or failed")
            }
        };
        JobOutcome {
            id: self.id,
            result,
            stored,
        }
    }
}

/// Runs `work` on every job, on up to `parallelism` threads at once, each
/// taking the next job as it finishes one: kernels differ by orders of
/// magnitude in how long they take to compile, so a fixed split would leave
/// threads idle behind the slowest share.
fn for_each_at_once<J: Send>(jobs: &mut [J], parallelism: usize, work: impl Fn(&mut J) + Sync) {
    #[cfg(all(feature = "std", not(target_family = "wasm")))]
    if jobs.len() > 1 && parallelism > 1 {
        use core::sync::atomic::{AtomicUsize, Ordering};

        let slots: Vec<std::sync::Mutex<&mut J>> =
            jobs.iter_mut().map(std::sync::Mutex::new).collect();
        let next = AtomicUsize::new(0);
        let panic = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..parallelism.min(slots.len()))
                .map(|_| {
                    scope.spawn(|| {
                        while let Some(slot) = slots.get(next.fetch_add(1, Ordering::Relaxed)) {
                            // Each slot is taken by exactly one thread, so the
                            // lock never waits; it only proves that to the
                            // compiler.
                            work(&mut slot.lock().unwrap_or_else(|err| err.into_inner()));
                        }
                    })
                })
                .collect();
            // Every worker is joined, so the scope itself never panics: left
            // to it, a worker's panic comes out as "a scoped thread panicked",
            // its message gone.
            workers
                .into_iter()
                .fold(None, |first, worker| first.or(worker.join().err()))
        });
        // The first panic goes on with its own payload, as it would have
        // on one thread.
        if let Some(payload) = panic {
            std::panic::resume_unwind(payload);
        }
        return;
    }

    let _ = parallelism;
    jobs.iter_mut().for_each(work);
}

/// Every test is serial: one that records reads the session every other one
/// would write its compilations to.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::CompilationError;
    use crate::id::KernelId;
    use crate::kernel::{KernelDefinition, KernelMetadata};
    use alloc::collections::BTreeSet;
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use cubecl_environment::backtrace::BackTrace;
    use std::sync::Mutex;

    /// Nothing compiles before something asks: queuing is free, and a load
    /// compiles the queue with the kernel it asked for.
    #[test]
    #[serial_test::serial(records)]
    fn a_load_compiles_the_queue() {
        let mut loader = loader(8);
        let logger = ServerLogger::default();
        for number in 0..3 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        assert_eq!(
            loader.target.compiler.lowered(),
            0,
            "queuing compiles nothing"
        );

        assert_eq!(loader.load(&Numbered(3), &id(3), &logger).unwrap(), 3);
        assert_eq!(
            loader.target.compiler.lowered(),
            4,
            "the queue compiled with it"
        );
        for number in 0..3 {
            assert_eq!(loader.get(&id(number)), Some(&number));
        }
    }

    /// Asked to, the queue compiles without a launch, and the launch that
    /// follows pays a lookup.
    #[test]
    #[serial_test::serial(records)]
    fn the_queue_compiles_when_asked() {
        let mut loader = loader(8);
        let logger = ServerLogger::default();
        for number in 0..3 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.compile_queued(&logger);
        assert_eq!(loader.target.compiler.lowered(), 3);
        assert_eq!(loader.load(&Numbered(0), &id(0), &logger).unwrap(), 0);
        assert_eq!(
            loader.target.compiler.lowered(),
            3,
            "nothing left to compile"
        );
    }

    /// A queue longer than the threads compiling it waits for its launch, and
    /// then compiles whole.
    #[test]
    #[serial_test::serial(records)]
    fn a_long_queue_compiles_in_one_phase() {
        let mut loader = loader(2);
        let logger = ServerLogger::default();
        for number in 0..5 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        assert_eq!(loader.target.compiler.lowered(), 0, "the queue waits");
        loader.load(&Numbered(5), &id(5), &logger).unwrap();
        assert_eq!(
            loader.target.compiler.lowered(),
            6,
            "with the kernel launched"
        );
    }

    /// A kernel queued twice, or queued once loaded, compiles once.
    #[test]
    #[serial_test::serial(records)]
    fn a_kernel_compiles_once() {
        let mut loader = loader(8);
        let logger = ServerLogger::default();
        loader.enqueue(Box::new(Numbered(0)), ());
        loader.load(&Numbered(0), &id(0), &logger).unwrap();
        loader.enqueue(Box::new(Numbered(0)), ());
        loader.load(&Numbered(0), &id(0), &logger).unwrap();
        assert_eq!(loader.target.compiler.lowered(), 1);
    }

    /// A queued kernel that fails does not take the others with it, and its
    /// launch reports the failure without compiling it again; the launch after
    /// that tries again.
    #[test]
    #[serial_test::serial(records)]
    fn a_failure_is_reported_by_its_own_launch() {
        let mut loader = loader(8);
        loader.target.compiler.failing = Some(1);
        let logger = ServerLogger::default();
        for number in 0..3 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.load(&Numbered(3), &id(3), &logger).unwrap();
        assert_eq!(loader.get(&id(0)), Some(&0));
        assert_eq!(loader.get(&id(1)), None);
        assert_eq!(loader.get(&id(2)), Some(&2));
        assert_eq!(loader.target.compiler.lowered(), 4);

        assert!(loader.load(&Numbered(1), &id(1), &logger).is_err());
        assert_eq!(
            loader.target.compiler.lowered(),
            4,
            "reported, not compiled again"
        );
        assert!(loader.load(&Numbered(1), &id(1), &logger).is_err());
        assert_eq!(
            loader.target.compiler.lowered(),
            5,
            "the next launch tries again"
        );
    }

    /// A failure the next batch finds unreported is dropped, not kept: the
    /// kernel's launch compiles it again.
    #[test]
    #[serial_test::serial(records)]
    fn a_stale_failure_compiles_again() {
        let mut loader = loader(8);
        loader.target.compiler.failing = Some(1);
        let logger = ServerLogger::default();
        loader.enqueue(Box::new(Numbered(1)), ());
        loader.load(&Numbered(0), &id(0), &logger).unwrap();
        loader.enqueue(Box::new(Numbered(2)), ());
        loader.load(&Numbered(0), &id(0), &logger).unwrap();
        assert_eq!(loader.target.compiler.lowered(), 3);

        loader.target.compiler.failing = None;
        assert_eq!(loader.load(&Numbered(1), &id(1), &logger).unwrap(), 1);
        assert_eq!(loader.target.compiler.lowered(), 4, "compiled again");
    }

    /// A kernel queued again after it failed is tried again.
    #[test]
    #[serial_test::serial(records)]
    fn a_failure_queued_again_is_retried() {
        let mut loader = loader(8);
        loader.target.compiler.failing = Some(1);
        let logger = ServerLogger::default();
        loader.enqueue(Box::new(Numbered(1)), ());
        loader.load(&Numbered(0), &id(0), &logger).unwrap();
        assert_eq!(loader.get(&id(1)), None);

        loader.target.compiler.failing = None;
        loader.enqueue(Box::new(Numbered(1)), ());
        loader.compile_queued(&logger);
        assert_eq!(loader.get(&id(1)), Some(&1));
    }

    /// A kernel that panics on a compiling thread panics the load with its
    /// own message, as it would on one thread.
    #[test]
    #[serial_test::serial(records)]
    fn a_panic_keeps_its_message() {
        let mut loader = loader(4);
        loader.target.compiler.panicking = Some(2);
        let logger = ServerLogger::default();
        for number in 0..4 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        let panic = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
            let _ = loader.load(&Numbered(4), &id(4), &logger);
        }))
        .expect_err("the load panics");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied());
        assert_eq!(message, Some("kernel 2 panicked"));
    }

    /// A batch is recorded once, with the wall time its kernels' work, spread
    /// over its threads, actually took.
    #[test]
    #[cfg(persistence)]
    #[serial_test::serial(records)]
    fn a_batch_is_recorded_beside_its_kernels() {
        use crate::compiler::{CompilationBatchRecord, CompilationRecord};
        use crate::config::RuntimeConfig;
        use cubecl_environment::persistence::Database;
        use cubecl_environment::records::{self, RecordLevel, Records, RecordsConfig};

        let root = tempfile::tempdir().unwrap();
        let _ = crate::config::CubeClRuntimeConfig::get();
        cubecl_environment::environment::set_root(root.path());
        records::configure(RecordsConfig {
            level: RecordLevel::Basic,
            ..Default::default()
        });

        let mut loader = loader(4);
        loader.target.compiler.slow = true;
        // A source the store takes: a session that changes nothing keeps
        // nothing it observed.
        loader.target.compiler.shared_source = true;
        let logger = ServerLogger::default();
        for number in 0..3 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.load(&Numbered(3), &id(3), &logger).unwrap();

        let database = Database::open_active().unwrap();
        let records = Records::new(&database);
        let batches = records.read::<CompilationBatchRecord>();
        let kernels = records.read::<CompilationRecord>();
        assert_eq!(batches.len(), 1);
        assert_eq!(kernels.len(), 4);
        records::configure(RecordsConfig {
            level: RecordLevel::Off,
            ..Default::default()
        });
        let batch = &batches[0].record;
        assert_eq!((batch.kernels, batch.threads), (4, 4));
        let work: core::time::Duration = kernels.iter().map(|kernel| kernel.record.work).sum();
        assert!(
            batch.wall < work,
            "four threads: {:?} of wall for {work:?} of work",
            batch.wall
        );
    }

    /// Kernels waiting on a source that fails to finalize fail with it: the
    /// source is finalized once, not once per kernel.
    #[test]
    #[serial_test::serial(records)]
    fn a_failed_source_is_finalized_once() {
        let mut loader = loader(8);
        loader.target.compiler.shared_source = true;
        loader.target.compiler.refusing = true;
        let logger = ServerLogger::default();
        for number in 0..3 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        assert!(loader.load(&Numbered(3), &id(3), &logger).is_err());
        assert_eq!(loader.target.compiler.finalized.load(Ordering::Relaxed), 1);
        for number in 0..3 {
            assert_eq!(loader.get(&id(number)), None);
        }
    }

    /// A kernel with no source to reuse an artifact by finalizes on the
    /// thread that lowered it, without waiting for the batch.
    #[test]
    #[serial_test::serial(records)]
    fn a_kernel_without_a_source_finalizes_where_it_lowered() {
        let mut loader = loader(4);
        loader.target.compiler.slow = true;
        let logger = ServerLogger::default();
        for number in 0..3 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.load(&Numbered(3), &id(3), &logger).unwrap();
        let lowered = loader.target.compiler.lowered_on.lock().unwrap().clone();
        let finalized = loader.target.compiler.finalized_on.lock().unwrap().clone();
        assert_eq!(lowered.len(), 4);
        assert_eq!(lowered, finalized);
    }

    /// A kernel about to execute compiles the queue even when it is loaded
    /// itself: the queue promises to be compiled by then.
    #[test]
    #[serial_test::serial(records)]
    fn a_loaded_kernel_still_compiles_the_queue() {
        let mut loader = loader(8);
        let logger = ServerLogger::default();
        loader.load(&Numbered(0), &id(0), &logger).unwrap();
        loader.enqueue(Box::new(Numbered(1)), ());
        loader.load(&Numbered(0), &id(0), &logger).unwrap();
        assert_eq!(loader.get(&id(1)), Some(&1));
    }

    /// Kernels of one batch that lower to the same source finalize it once:
    /// the others take the artifact the first one stored.
    #[test]
    #[serial_test::serial(records)]
    fn one_source_finalizes_once() {
        let mut loader = loader(8);
        loader.target.compiler.shared_source = true;
        let logger = ServerLogger::default();
        for number in 0..4 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.load(&Numbered(4), &id(4), &logger).unwrap();
        assert_eq!(loader.target.compiler.finalized.load(Ordering::Relaxed), 1);
        for number in 0..5 {
            assert!(loader.get(&id(number)).is_some());
        }
    }

    /// A source the store already holds serves every kernel of a batch that
    /// lowers to it: none of them finalizes it again.
    #[test]
    #[serial_test::serial(records)]
    fn a_stored_source_serves_the_whole_batch() {
        let mut loader = loader(8);
        loader.target.compiler.shared_source = true;
        loader.target.by_source.insert("shared".into(), 99);
        let logger = ServerLogger::default();
        for number in 0..2 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.load(&Numbered(2), &id(2), &logger).unwrap();
        assert_eq!(loader.target.compiler.finalized.load(Ordering::Relaxed), 0);
        for number in 0..3 {
            assert_eq!(loader.get(&id(number)), Some(&99));
        }
    }

    /// Without a store there is no artifact to wait for: every kernel
    /// finalizes its own, on the compiling threads.
    #[test]
    #[serial_test::serial(records)]
    fn without_a_store_each_kernel_finalizes_its_own() {
        let mut loader = KernelLoader::new(Fake {
            storeless: true,
            ..Fake::default()
        });
        loader.parallelism = 8;
        loader.target.compiler.shared_source = true;
        let logger = ServerLogger::default();
        for number in 0..3 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.load(&Numbered(3), &id(3), &logger).unwrap();
        assert_eq!(loader.target.compiler.finalized.load(Ordering::Relaxed), 4);
    }

    /// A queue is lowered on several threads at once.
    #[test]
    #[serial_test::serial(records)]
    fn a_queue_compiles_on_several_threads() {
        let mut loader = loader(4);
        loader.target.compiler.slow = true;
        let logger = ServerLogger::default();
        for number in 0..8 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.load(&Numbered(8), &id(8), &logger).unwrap();
        let threads = loader.target.compiler.threads.lock().unwrap().len();
        assert!(threads > 1, "lowered on {threads} thread");
    }

    fn loader(parallelism: usize) -> KernelLoader<Fake> {
        let mut loader = KernelLoader::new(Fake::default());
        loader.parallelism = parallelism;
        loader
    }

    fn id(number: u32) -> ArtifactId<()> {
        ArtifactId {
            kernel: Numbered(number).id(),
            variant: (),
        }
    }

    /// A kernel the fake compiler tells apart by its number, and never
    /// expands.
    #[derive(Debug)]
    struct Numbered(u32);

    impl KernelMetadata for Numbered {
        fn id(&self) -> KernelId {
            KernelId::new::<Self>().info(self.0)
        }

        fn address_type(&self) -> cubecl_ir::ElemType {
            cubecl_ir::ElemType::UInt(cubecl_ir::UIntKind::U32)
        }
    }

    impl CubeKernel for Numbered {
        fn define(&self) -> KernelDefinition {
            unreachable!("the fake compiler reads the number, not the definition")
        }
    }

    /// A target whose kernels are their numbers, keeping its artifacts by
    /// source only.
    #[derive(Debug, Default)]
    struct Fake {
        compiler: FakeCompiler,
        by_source: alloc::collections::BTreeMap<String, u32>,
        /// Whether it keeps nothing between runs, by source or otherwise.
        storeless: bool,
    }

    #[derive(Debug, Default)]
    struct FakeCompiler {
        lowered: AtomicUsize,
        /// The number of the kernel that fails to lower.
        failing: Option<u32>,
        /// The number of the kernel whose lowering panics.
        panicking: Option<u32>,
        /// Whether lowering takes long enough for other threads to pick up
        /// the rest of the queue.
        slow: bool,
        /// The threads lowering ran on.
        threads: Mutex<BTreeSet<String>>,
        /// Whether every kernel lowers to the same source.
        shared_source: bool,
        finalized: AtomicUsize,
        /// Whether finalizing fails.
        refusing: bool,
        /// The thread each kernel lowered on, by number.
        lowered_on: Mutex<alloc::collections::BTreeMap<u32, String>>,
        /// The thread each kernel finalized on, by number.
        finalized_on: Mutex<alloc::collections::BTreeMap<u32, String>>,
    }

    impl FakeCompiler {
        fn lowered(&self) -> usize {
            self.lowered.load(Ordering::Relaxed)
        }
    }

    impl ArtifactCompiler for FakeCompiler {
        type Variant = ();
        type Lowered = (u32, Option<String>);
        type Artifact = u32;

        fn lower(
            &self,
            kernel: &dyn CubeKernel,
            id: &ArtifactId<()>,
            _recording: &mut CompilationRecording,
            _logger: &ServerLogger,
        ) -> Result<(u32, Option<String>), LaunchError> {
            self.lowered.fetch_add(1, Ordering::Relaxed);
            self.threads
                .lock()
                .unwrap()
                .insert(alloc::format!("{:?}", std::thread::current().id()));
            if self.slow {
                std::thread::sleep(core::time::Duration::from_millis(20));
            }

            let number = (0..64)
                .find(|number| Numbered(*number).id() == id.kernel)
                .expect("a numbered kernel");
            assert_eq!(kernel.id(), id.kernel);
            self.lowered_on
                .lock()
                .unwrap()
                .insert(number, alloc::format!("{:?}", std::thread::current().id()));
            if self.panicking == Some(number) {
                panic!("kernel {number} panicked");
            }
            match self.failing == Some(number) {
                true => Err(LaunchError::Unknown {
                    reason: "refused".into(),
                    backtrace: BackTrace::capture(),
                }),
                false => Ok((number, self.shared_source.then(|| "shared".into()))),
            }
        }

        fn source(lowered: &(u32, Option<String>)) -> Option<&str> {
            lowered.1.as_deref()
        }

        fn finalize(
            &self,
            _id: &ArtifactId<()>,
            lowered: (u32, Option<String>),
        ) -> Result<u32, LaunchError> {
            self.finalized.fetch_add(1, Ordering::Relaxed);
            self.finalized_on.lock().unwrap().insert(
                lowered.0,
                alloc::format!("{:?}", std::thread::current().id()),
            );
            match self.refusing {
                true => Err(LaunchError::Unknown {
                    reason: "refused".into(),
                    backtrace: BackTrace::capture(),
                }),
                false => Ok(lowered.0),
            }
        }
    }

    impl CompilationTarget for Fake {
        type Compiler = FakeCompiler;
        type Loaded = u32;

        fn compiler(&self) -> &FakeCompiler {
            &self.compiler
        }

        fn persists(&self) -> bool {
            !self.storeless
        }

        fn stored(&mut self, _id: &ArtifactId<()>) -> Option<u32> {
            None
        }

        fn load(&mut self, _id: &ArtifactId<()>, artifact: &u32) -> Result<u32, CompilationError> {
            Ok(*artifact)
        }

        fn stored_for_source(&mut self, source: &str) -> Option<u32> {
            self.by_source.remove(source)
        }

        fn store(&mut self, _id: &ArtifactId<()>, artifact: u32, source: Option<&str>) -> bool {
            if let Some(source) = source {
                self.by_source.insert(source.into(), artifact);
            }
            source.is_some()
        }
    }
}
