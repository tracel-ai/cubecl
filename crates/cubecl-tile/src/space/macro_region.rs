use crate::{Dimension, MicroRegion, Space};

/// The smallest region that a partitioning can count in: the one the [`MicroRegion`] and the
/// region of the instruction divide, so inside it every read, every write and every instruction is
/// whole.
///
/// Under an instruction of `{m 8, n 8, k 8}`, reads whose micro region is `{k 64}` have the macro
/// region `{m 8, n 8, k 64}`, eight instructions deep. Where the two divide well it is the larger
/// of them.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MacroRegion {
    instruction: Space,
    instructions: Space,
}

impl MacroRegion {
    /// The macro region of reads and writes whose micro region is `micro`, under an instruction
    /// that computes `instruction` at once.
    ///
    /// # Panics
    ///
    /// On a dynamic extent, which has no multiple.
    pub fn new(micro: &MicroRegion, instruction: Space) -> Self {
        Self::counting(instruction, &[micro.space().clone()])
    }

    /// This region, `moved` dividing it too: what an engine other than the units moves whole.
    ///
    /// # Panics
    ///
    /// On a dynamic extent, which has no multiple.
    pub fn divided_by(&self, moved: Space) -> Self {
        Self::counting(self.instruction.clone(), &[self.space(), moved])
    }

    /// How far the region reaches along `dimension`.
    pub fn along(&self, dimension: Dimension) -> usize {
        self.instruction.along(dimension) * self.instructions.along(dimension)
    }

    /// The region of one instruction.
    pub fn instruction(&self) -> &Space {
        &self.instruction
    }

    /// The instructions the region counts along `dimension`.
    pub fn instructions(&self, dimension: Dimension) -> usize {
        self.instructions.along(dimension)
    }

    /// The region as a box of its dimensions.
    pub fn space(&self) -> Space {
        Space::new(
            &self
                .instructions
                .dimensions()
                .map(|dimension| (dimension, self.along(dimension)))
                .collect::<Vec<_>>(),
        )
    }
}

impl MacroRegion {
    /// The fewest `instruction`s every one of `divisors` divides.
    fn counting(instruction: Space, divisors: &[Space]) -> Self {
        Self {
            instructions: instruction.counts_whole_for(divisors),
            instruction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Dimension = Dimension(0);
    const N: Dimension = Dimension(1);
    const K: Dimension = Dimension(2);

    fn under(instruction: [usize; 3]) -> MacroRegion {
        let micro = MicroRegion::new([Space::new(&[(K, 32), (N, 16)]), Space::new(&[(M, 4)])]);
        let [m, n, k] = instruction;
        MacroRegion::new(&micro, Space::new(&[(M, m), (N, n), (K, k)]))
    }

    /// Where the micro region and the instruction divide well, the region is the larger of them
    /// along each dimension.
    #[test]
    fn a_region_is_the_larger_where_both_divide_well() {
        let region = under([16, 16, 16]);
        assert_eq!(
            [M, N, K].map(|dimension| region.along(dimension)),
            [16, 16, 32]
        );
    }

    /// The region counts the instructions it is made of along each dimension.
    #[test]
    fn a_region_counts_its_instructions() {
        let region = under([16, 16, 16]);
        assert_eq!(
            [M, N, K].map(|dimension| region.instructions(dimension)),
            [1, 1, 2]
        );
    }

    /// An instruction that does not nest with the reads meets them at their least common multiple.
    #[test]
    fn an_instruction_that_does_not_nest_meets_the_reads_at_their_multiple() {
        assert_eq!(under([7, 7, 7]).along(K), 224);
    }

    /// The region reaches along a dimension the reads name and the instruction does not.
    #[test]
    fn a_region_reaches_along_a_dimension_only_the_reads_name() {
        let micro = MicroRegion::new([Space::new(&[(K, 32)])]);
        let region = MacroRegion::new(&micro, Space::new(&[(M, 8)]));
        assert_eq!((region.along(M), region.along(K)), (8, 32));
    }

    /// What another engine moves whole divides the region too, its instructions counted again.
    #[test]
    fn what_an_engine_moves_whole_divides_the_region() {
        let region = under([16, 16, 16]).divided_by(Space::new(&[(K, 128)]));
        assert_eq!((region.along(K), region.instructions(K)), (128, 8));
    }
}
