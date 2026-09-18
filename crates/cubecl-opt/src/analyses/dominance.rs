use pliron::{
    graph::{
        ControlFlowGraph,
        dominance::{DomFrontierMap, DomTree},
    },
    utils::table::ISet,
};
use smallvec::SmallVec;

pub struct DomFrontierCalculator<G: ControlFlowGraph<GraphContext>, GraphContext> {
    frontiers: DomFrontierMap<G, GraphContext>,
    defining_blocks: SmallVec<[G::Node; 16]>,
}

impl<G: ControlFlowGraph<GraphContext>, GraphContext> DomFrontierCalculator<G, GraphContext> {
    pub fn new(
        ctx: &GraphContext,
        graph: &G,
        tree: &DomTree<G, GraphContext>,
        defining_blocks: impl IntoIterator<Item = G::Node>,
    ) -> Self {
        let frontiers = DomFrontierMap::new(ctx, graph, tree);
        Self {
            frontiers,
            defining_blocks: defining_blocks.into_iter().collect(),
        }
    }

    pub fn compute(&self) -> ISet<G::Node> {
        let mut worklist = self.defining_blocks.clone();
        let mut merge_points = ISet::default();

        while let Some(block) = worklist.pop() {
            for frontier_block in self.frontiers.frontier(&block) {
                if merge_points.insert(frontier_block.clone()) {
                    worklist.push(frontier_block.clone());
                }
            }
        }

        merge_points
    }
}
