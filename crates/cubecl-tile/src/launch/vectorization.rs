use crate::layout::ElemAlignment;

/// Why an operand's width must be narrower than the device reads it at.
///
/// A requirement is about the tiles of an operand a component touches, never about the memory
/// they live in: a microkernel is handed tiles and does not know whether its walker staged them.
/// That holds because an operand keeps one width down the whole chain — the global buffer is
/// bound at it, a stage the ring fills keeps it, a fragment loads the stage at it — so every
/// requirement on an operand, from every component, narrows the same width.
///
/// A requirement's elements lie along the operand's contiguous axis, and a stretch starts at a
/// multiple of its own length, since every tile is a regular cut of its parent: a width dividing
/// the length never has a line cross the end of a stretch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorRequirement<O> {
    /// A component reads or writes `operand`'s tiles in stretches of `elements` consecutive
    /// elements, so the width divides them.
    Access { operand: O, elements: usize },
    /// A component reads or writes `operand` one element at a time, as out of shared-memory
    /// atomics, which hand back a scalar: the width is one.
    Elementwise { operand: O },
    /// The launch binds `operand` masked one element at a time, as it does where a tile reaches
    /// past the operand's edge along its contiguous axis: a line in the last, short tile would
    /// reach past it, so the width is one.
    Masked { operand: O },
    /// A component lines `operand` together with `with`, one line of each per step, as a register
    /// block lines the rhs along `n` with its accumulator: the two share one width, the narrowest
    /// of what every operand lined with either admits alone.
    Together { operand: O, with: O },
    /// The launch binds `operand`'s buffer, whose layout is aligned to `alignment` elements along
    /// its innermost dim: a line must read its elements and no others.
    Aligned {
        operand: O,
        alignment: ElemAlignment,
    },
}

impl<O: Copy + Eq> VectorRequirement<O> {
    /// The operand this requirement narrows.
    pub fn operand(&self) -> O {
        match self {
            VectorRequirement::Access { operand, .. }
            | VectorRequirement::Elementwise { operand }
            | VectorRequirement::Masked { operand }
            | VectorRequirement::Together { operand, .. }
            | VectorRequirement::Aligned { operand, .. } => *operand,
        }
    }

    /// This requirement on the operands `operand` maps its own onto, where it maps every one: what
    /// a family running a shared microkernel on its own operands requires of them. `None` for an
    /// operand the family binds no buffer for, as a product it keeps in registers.
    pub fn restated<P>(self, operand: impl Fn(O) -> Option<P>) -> Option<VectorRequirement<P>> {
        Some(match self {
            VectorRequirement::Access {
                operand: at,
                elements,
            } => VectorRequirement::Access {
                operand: operand(at)?,
                elements,
            },
            VectorRequirement::Elementwise { operand: at } => VectorRequirement::Elementwise {
                operand: operand(at)?,
            },
            VectorRequirement::Masked { operand: at } => VectorRequirement::Masked {
                operand: operand(at)?,
            },
            VectorRequirement::Together { operand: at, with } => VectorRequirement::Together {
                operand: operand(at)?,
                with: operand(with)?,
            },
            VectorRequirement::Aligned {
                operand: at,
                alignment,
            } => VectorRequirement::Aligned {
                operand: operand(at)?,
                alignment,
            },
        })
    }

    /// Whether a line of `width` meets this requirement.
    pub fn admits(&self, width: usize) -> bool {
        match self {
            VectorRequirement::Access { elements, .. } => elements.is_multiple_of(width),
            VectorRequirement::Elementwise { .. } | VectorRequirement::Masked { .. } => width == 1,
            VectorRequirement::Aligned { alignment, .. } => alignment.admits(width),
            // Settled across the operands it lines together by `width`, not by either alone.
            VectorRequirement::Together { .. } => true,
        }
    }
}

/// The requirements every component of a launch enqueued, and the widths they settle.
///
/// A routine starts the pass with its own accesses and hands it to its microkernel, which adds
/// its own and hands it to the ones it is built from: the pass walks down as the kernel does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vectorization<O> {
    requirements: Vec<VectorRequirement<O>>,
}

impl<O: Copy + Eq> Vectorization<O> {
    pub fn new() -> Self {
        Self {
            requirements: Vec::new(),
        }
    }

    pub fn require(&mut self, requirement: VectorRequirement<O>) {
        self.requirements.push(requirement);
    }

    /// Every requirement `shared` holds, restated on this pass's operands through `operand`
    /// ([`VectorRequirement::restated`]): how a family hands a shared microkernel its own pass,
    /// one multiplication's operands at a time.
    pub fn restate<S: Copy + Eq>(
        &mut self,
        shared: Vectorization<S>,
        operand: impl Fn(S) -> Option<O>,
    ) {
        let restated = shared
            .requirements
            .into_iter()
            .filter_map(|requirement| requirement.restated(&operand));
        self.requirements.extend(restated);
    }

    /// The width `operand` binds at: the narrowest, over `operand` and every operand lined together
    /// with it through [`Together`](VectorRequirement::Together) pairs, of the widest power of two
    /// at most `widest(member)` every requirement on that member admits.
    pub fn width(&self, operand: O, widest: impl Fn(O) -> usize) -> usize {
        let mut lined = vec![operand];
        let mut next = 0;
        while let Some(&at) = lined.get(next) {
            for requirement in &self.requirements {
                let VectorRequirement::Together { operand, with } = *requirement else {
                    continue;
                };
                let other = match (operand == at, with == at) {
                    (true, _) => with,
                    (_, true) => operand,
                    (false, false) => continue,
                };
                if !lined.contains(&other) {
                    lined.push(other);
                }
            }
            next += 1;
        }
        lined
            .into_iter()
            .map(|member| self.alone(member, widest(member)))
            .min()
            .unwrap_or(1)
    }

    /// The widest power of two, at most `widest`, every requirement on `operand` admits.
    fn alone(&self, operand: O, widest: usize) -> usize {
        let admitted = |width: usize| {
            self.requirements
                .iter()
                .filter(|requirement| requirement.operand() == operand)
                .all(|requirement| requirement.admits(width))
        };
        let mut width = widest.max(1).next_power_of_two();
        while width > 1 && (width > widest || !admitted(width)) {
            width /= 2;
        }
        width
    }
}

impl<O: Copy + Eq> Default for Vectorization<O> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Operands lined together share one width, the narrowest any of them admits alone — through a
    /// chain of pairs too; an operand lined with none keeps its own.
    #[test]
    fn operands_lined_together_share_the_narrowest_width() {
        let mut vectorization = Vectorization::new();
        vectorization.require(VectorRequirement::Access {
            operand: 'c',
            elements: 2,
        });
        vectorization.require(VectorRequirement::Together {
            operand: 'a',
            with: 'b',
        });
        vectorization.require(VectorRequirement::Together {
            operand: 'b',
            with: 'c',
        });
        for operand in ['a', 'b', 'c'] {
            assert_eq!(vectorization.width(operand, |_| 8), 2, "{operand}");
        }
        assert_eq!(vectorization.width('d', |_| 8), 8);
    }

    /// A buffer aligned to fewer elements than the device reads narrows the width to what it
    /// admits.
    #[test]
    fn the_width_reads_the_buffer_whole_lines() {
        let mut vectorization = Vectorization::new();
        vectorization.require(VectorRequirement::Aligned {
            operand: 'a',
            alignment: ElemAlignment::new(&[4, 12], &[12, 1]),
        });
        assert_eq!(vectorization.width('a', |_| 8), 4);
        vectorization.require(VectorRequirement::Aligned {
            operand: 'b',
            alignment: ElemAlignment::ONE,
        });
        assert_eq!(vectorization.width('b', |_| 8), 1);
    }

    /// A shared microkernel's requirements land on the operands the family binds; one on an
    /// operand it keeps in registers, and a pair lined with one, land nowhere.
    #[test]
    fn a_restated_requirement_lands_on_the_bound_operand() {
        let mut shared = Vectorization::new();
        shared.require(VectorRequirement::Access {
            operand: 'l',
            elements: 4,
        });
        shared.require(VectorRequirement::Access {
            operand: 'o',
            elements: 1,
        });
        shared.require(VectorRequirement::Together {
            operand: 'r',
            with: 'o',
        });
        let mut family = Vectorization::new();
        family.restate(shared, |operand| match operand {
            'l' => Some('a'),
            'r' => Some('b'),
            _ => None,
        });
        assert_eq!(family.width('a', |_| 8), 4);
        assert_eq!(family.width('b', |_| 8), 8);
    }

    #[test]
    fn the_width_divides_every_access() {
        let mut vectorization = Vectorization::new();
        vectorization.require(VectorRequirement::Access {
            operand: 'a',
            elements: 24,
        });
        assert_eq!(vectorization.width('a', |_| 16), 8);
        vectorization.require(VectorRequirement::Access {
            operand: 'a',
            elements: 12,
        });
        assert_eq!(vectorization.width('a', |_| 16), 4);
        assert_eq!(vectorization.width('b', |_| 16), 16);
    }

    #[test]
    fn an_elementwise_access_is_scalar() {
        let mut vectorization = Vectorization::new();
        vectorization.require(VectorRequirement::Access {
            operand: 'a',
            elements: 8,
        });
        vectorization.require(VectorRequirement::Elementwise { operand: 'a' });
        assert_eq!(vectorization.width('a', |_| 8), 1);
    }

    #[test]
    fn the_width_never_exceeds_the_widest_it_is_given() {
        let vectorization = Vectorization::<char>::new();
        assert_eq!(vectorization.width('a', |_| 2), 2);
        assert_eq!(vectorization.width('a', |_| 3), 2);
        assert_eq!(vectorization.width('a', |_| 1), 1);
    }
}
