//! The kernels a device has loaded, and the one path that loads more.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use cubecl_common::profile::Instant;
use cubecl_environment::collections::{HashMap, HashSet};

use super::{
    ArtifactCompiler, ArtifactId, CompilationCache, CompilationOutcome, CompilationRecording,
    CompilationTarget, VariantOf,
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
    /// The kernels queued and not compiled yet, each once.
    queue: Vec<Queued<VariantOf<T>>>,
    /// The ids of [`queue`](Self::queue)'s kernels, so queuing one more costs
    /// a lookup rather than a scan.
    queued: HashSet<ArtifactId<VariantOf<T>>>,
    /// Why each queued kernel that failed to compile failed, until its own
    /// launch reports it: the launch is where the failure has outputs to
    /// carry, and compiling the kernel again there would only fail again.
    failed: HashMap<ArtifactId<VariantOf<T>>, LaunchError>,
    /// How many threads compile at once.
    parallelism: usize,
}

impl<T: CompilationTarget> KernelLoader<T> {
    /// A loader holding nothing yet, bound to the active environment exactly
    /// when `target` [persists](CompilationTarget::persists) what it compiles.
    pub fn new(target: T) -> Self {
        let loaded = match target.persists() {
            true => CompilationCache::bound(),
            false => CompilationCache::unbound(),
        };
        let parallelism = crate::config::CubeClRuntimeConfig::get()
            .compilation
            .parallelism();

        Self {
            target,
            loaded,
            queue: Vec::new(),
            queued: HashSet::new(),
            failed: HashMap::new(),
            parallelism,
        }
    }

    /// The backend this loader compiles through.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// The kernel loaded for `id`, if it is.
    pub fn get(&mut self, id: &ArtifactId<VariantOf<T>>) -> Option<&T::Loaded> {
        self.loaded.get(id)
    }

    /// The kernel loaded for `kernel`, under `id`, loading it first if it is
    /// not. A kernel is about to execute, so every queued kernel is compiled
    /// first — together with this one when it needs compiling.
    ///
    /// `id` is `kernel`'s, with what the launch adds to it: the launch already
    /// built the kernel's id, and building it again on every launch costs.
    ///
    /// # Errors
    ///
    /// The validation, compilation or load of `kernel` that failed, now or
    /// when it was queued. The failure is reported once and not kept, so the
    /// next launch of the kernel tries again.
    pub fn load(
        &mut self,
        kernel: &dyn CubeKernel,
        id: ArtifactId<VariantOf<T>>,
        logger: &ServerLogger,
    ) -> Result<&T::Loaded, LaunchError> {
        let missing = !self.loaded.contains(&id) && !self.failed.contains_key(&id);
        if missing || !self.queue.is_empty() {
            let queue = core::mem::take(&mut self.queue);
            self.queued.clear();
            let mut requests: Vec<Request<'_, VariantOf<T>>> = queue
                .iter()
                .filter(|queued| queued.id != id)
                .map(Queued::request)
                .collect();
            if missing {
                requests.push(Request {
                    id: id.clone(),
                    kernel,
                });
            }
            for compiled in self.compile(requests, logger) {
                self.keep(compiled);
            }
        }

        if let Some(err) = self.failed.remove(&id) {
            return Err(err);
        }
        // An environment switch during the batch drops every kernel loaded
        // before it, this one too when it was: it loads again.
        if !self.loaded.contains(&id) {
            return self.load(kernel, id, logger);
        }
        Ok(self.loaded.get(&id).expect("checked right above"))
    }

    /// Queues `kernel` under `variant`, to be compiled with the others when the
    /// next kernel is [loaded](Self::load) for a launch, and only then. One
    /// already loaded, queued, or failed and not yet reported is left alone.
    ///
    /// The queue does not compile on its own when it grows: one phase over
    /// everything queued lasts about as long as its slowest kernel, where
    /// two phases would each wait on theirs.
    pub fn enqueue(&mut self, kernel: Box<dyn CubeKernel>, variant: VariantOf<T>) {
        let id = ArtifactId {
            kernel: kernel.id(),
            variant,
        };
        if self.loaded.contains(&id) || self.failed.contains_key(&id) || self.queued.contains(&id) {
            return;
        }

        self.queued.insert(id.clone());
        self.queue.push(Queued { id, kernel });
    }

    /// Keeps a compiled kernel, or the reason it failed.
    fn keep(&mut self, compiled: Compiled<T>) {
        match compiled.result {
            Ok(loaded) => self.loaded.insert(compiled.id, loaded),
            Err(err) => {
                self.failed.insert(compiled.id, err);
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
    ) -> Vec<Compiled<T>> {
        let mut jobs: Vec<Job<'_, T>> = requests.into_iter().map(Job::new).collect();
        let jobs_len = jobs.len();

        for job in jobs.iter_mut() {
            job.look_up(&mut self.target);
        }

        let compiler = self.target.compiler();
        let mut missing: Vec<&mut Job<'_, T>> =
            jobs.iter_mut().filter(|job| job.is_missing()).collect();
        if missing.len() > 1 {
            log::info!(
                target: "cubecl::compilation",
                "Compiling {} kernels together ({} asked, the rest read from the store) on {} threads",
                missing.len(),
                jobs_len,
                self.parallelism.min(missing.len()),
            );
        }
        for_each_at_once(&mut missing, self.parallelism, |job| {
            job.lower(compiler, logger)
        });

        let mut finalizing = HashSet::new();
        for job in jobs.iter_mut() {
            job.reuse(&mut self.target, &mut finalizing);
        }

        let compiler = self.target.compiler();
        let mut lowered: Vec<&mut Job<'_, T>> =
            jobs.iter_mut().filter(|job| job.is_lowered()).collect();
        for_each_at_once(&mut lowered, self.parallelism, |job| job.finalize(compiler));

        jobs.into_iter()
            .map(|job| job.load(&mut self.target))
            .collect()
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

/// A kernel compiled and loaded, or why it was not.
struct Compiled<T: CompilationTarget> {
    id: ArtifactId<VariantOf<T>>,
    result: Result<T::Loaded, LaunchError>,
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
    fn load(mut self, target: &mut T) -> Compiled<T> {
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

        let result =
            match self.step {
                Step::Ready {
                    artifact,
                    source,
                    outcome,
                } => {
                    let started = Instant::now();
                    match target.load(&self.id, &artifact) {
                        Ok(loaded) => {
                            self.recording.worked(started.elapsed());
                            match outcome {
                                CompilationOutcome::Loaded => self.recording.loaded(),
                                CompilationOutcome::Compiled => self
                                    .recording
                                    .compiled(target.store(&self.id, artifact, source.as_deref())),
                                CompilationOutcome::Rekeyed => self
                                    .recording
                                    .rekeyed(target.store(&self.id, artifact, source.as_deref())),
                            }
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
        Compiled {
            id: self.id,
            result,
        }
    }
}

/// Runs `work` on every job, on up to `parallelism` threads at once, each
/// taking the next job as it finishes one: kernels differ by orders of
/// magnitude in how long they take to compile, so a fixed split would leave
/// threads idle behind the slowest share.
fn for_each_at_once<J: Send>(jobs: &mut [J], parallelism: usize, work: impl Fn(&mut J) + Sync) {
    #[cfg(feature = "std")]
    if jobs.len() > 1 && parallelism > 1 {
        use core::sync::atomic::{AtomicUsize, Ordering};

        let slots: Vec<std::sync::Mutex<&mut J>> =
            jobs.iter_mut().map(std::sync::Mutex::new).collect();
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..parallelism.min(slots.len()) {
                scope.spawn(|| {
                    while let Some(slot) = slots.get(next.fetch_add(1, Ordering::Relaxed)) {
                        // Each slot is taken by exactly one thread, so the lock
                        // never waits; it only proves that to the compiler.
                        work(&mut slot.lock().unwrap_or_else(|err| err.into_inner()));
                    }
                });
            }
        });
        return;
    }

    let _ = parallelism;
    jobs.iter_mut().for_each(work);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::CompilationError;
    use crate::id::KernelId;
    use crate::kernel::{KernelDefinition, KernelMetadata};
    use alloc::collections::BTreeSet;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use cubecl_environment::backtrace::BackTrace;
    use std::sync::Mutex;

    /// Nothing compiles before something asks: queuing is free, and a load
    /// compiles the queue with the kernel it asked for.
    #[test]
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

        assert_eq!(*loader.load(&Numbered(3), id(3), &logger).unwrap(), 3);
        assert_eq!(
            loader.target.compiler.lowered(),
            4,
            "the queue compiled with it"
        );
        for number in 0..3 {
            assert_eq!(loader.get(&id(number)), Some(&number));
        }
    }

    /// A queue longer than the threads compiling it waits for its launch, and
    /// then compiles whole.
    #[test]
    fn a_long_queue_compiles_in_one_phase() {
        let mut loader = loader(2);
        let logger = ServerLogger::default();
        for number in 0..5 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        assert_eq!(loader.target.compiler.lowered(), 0, "the queue waits");
        loader.load(&Numbered(5), id(5), &logger).unwrap();
        assert_eq!(
            loader.target.compiler.lowered(),
            6,
            "with the kernel launched"
        );
    }

    /// A kernel queued twice, or queued once loaded, compiles once.
    #[test]
    fn a_kernel_compiles_once() {
        let mut loader = loader(8);
        let logger = ServerLogger::default();
        loader.enqueue(Box::new(Numbered(0)), ());
        loader.load(&Numbered(0), id(0), &logger).unwrap();
        loader.enqueue(Box::new(Numbered(0)), ());
        loader.load(&Numbered(0), id(0), &logger).unwrap();
        assert_eq!(loader.target.compiler.lowered(), 1);
    }

    /// A queued kernel that fails does not take the others with it, and its
    /// launch reports the failure without compiling it again; the launch after
    /// that tries again.
    #[test]
    fn a_failure_is_reported_by_its_own_launch() {
        let mut loader = loader(8);
        loader.target.compiler.failing = Some(1);
        let logger = ServerLogger::default();
        for number in 0..3 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.load(&Numbered(3), id(3), &logger).unwrap();
        assert_eq!(loader.get(&id(0)), Some(&0));
        assert_eq!(loader.get(&id(1)), None);
        assert_eq!(loader.get(&id(2)), Some(&2));
        assert_eq!(loader.target.compiler.lowered(), 4);

        assert!(loader.load(&Numbered(1), id(1), &logger).is_err());
        assert_eq!(
            loader.target.compiler.lowered(),
            4,
            "reported, not compiled again"
        );
        assert!(loader.load(&Numbered(1), id(1), &logger).is_err());
        assert_eq!(
            loader.target.compiler.lowered(),
            5,
            "the next launch tries again"
        );
    }

    /// A kernel about to execute compiles the queue even when it is loaded
    /// itself: the queue promises to be compiled by then.
    #[test]
    fn a_loaded_kernel_still_compiles_the_queue() {
        let mut loader = loader(8);
        let logger = ServerLogger::default();
        loader.load(&Numbered(0), id(0), &logger).unwrap();
        loader.enqueue(Box::new(Numbered(1)), ());
        loader.load(&Numbered(0), id(0), &logger).unwrap();
        assert_eq!(loader.get(&id(1)), Some(&1));
    }

    /// Kernels of one batch that lower to the same source finalize it once:
    /// the others take the artifact the first one stored.
    #[test]
    fn one_source_finalizes_once() {
        let mut loader = loader(8);
        loader.target.compiler.shared_source = true;
        let logger = ServerLogger::default();
        for number in 0..4 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.load(&Numbered(4), id(4), &logger).unwrap();
        assert_eq!(loader.target.compiler.finalized.load(Ordering::Relaxed), 1);
        for number in 0..5 {
            assert!(loader.get(&id(number)).is_some());
        }
    }

    /// A source the store already holds serves every kernel of a batch that
    /// lowers to it: none of them finalizes it again.
    #[test]
    fn a_stored_source_serves_the_whole_batch() {
        let mut loader = loader(8);
        loader.target.compiler.shared_source = true;
        loader.target.by_source.insert("shared".into(), 99);
        let logger = ServerLogger::default();
        for number in 0..2 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.load(&Numbered(2), id(2), &logger).unwrap();
        assert_eq!(loader.target.compiler.finalized.load(Ordering::Relaxed), 0);
        for number in 0..3 {
            assert_eq!(loader.get(&id(number)), Some(&99));
        }
    }

    /// Without a store there is no artifact to wait for: every kernel
    /// finalizes its own, on the compiling threads.
    #[test]
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
        loader.load(&Numbered(3), id(3), &logger).unwrap();
        assert_eq!(loader.target.compiler.finalized.load(Ordering::Relaxed), 4);
    }

    /// A queue is lowered on several threads at once.
    #[test]
    fn a_queue_compiles_on_several_threads() {
        let mut loader = loader(4);
        loader.target.compiler.slow = true;
        let logger = ServerLogger::default();
        for number in 0..8 {
            loader.enqueue(Box::new(Numbered(number)), ());
        }
        loader.load(&Numbered(8), id(8), &logger).unwrap();
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
        /// Whether lowering takes long enough for other threads to pick up
        /// the rest of the queue.
        slow: bool,
        /// The threads lowering ran on.
        threads: Mutex<BTreeSet<String>>,
        /// Whether every kernel lowers to the same source.
        shared_source: bool,
        finalized: AtomicUsize,
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
            Ok(lowered.0)
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
