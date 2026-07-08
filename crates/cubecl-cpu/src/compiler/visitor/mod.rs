pub(super) mod args_manager;
pub(super) mod block;
pub(super) mod elem;
pub(super) mod instruction;
pub(super) mod item;
pub(super) mod operation;
pub(super) mod prelude;
pub(super) mod values;

use std::collections::HashMap;

use args_manager::{ArgsManager, ArgsManagerBuilder};
use cubecl_core::{
    ir::{self as cube, Builtin, Id, StorageType},
    prelude::KernelDefinition,
};
use cubecl_opt::{Function, GlobalState, MemoryLiveness, NodeIndex};
use std::rc::Rc;
use tracel_llvm::mlir_rs::{
    Context,
    dialect::{
        func,
        llvm::{
            self,
            attributes::{Linkage, linkage},
        },
        ods::llvm as llvm_ods,
        scf,
    },
    ir::{
        Attribute, Block, BlockRef, Identifier, Location, Module, Operation, Region, RegionRef,
        Value,
        attribute::{StringAttribute, TypeAttribute},
        r#type::IntegerType,
    },
};

use prelude::*;
use values::Values;

use crate::compiler::visitor::operation::synchronization::add_sync_cube_function;

use super::{
    external_function::add_external_function_to_module, passes::shared_memories::SharedMemories,
};

pub struct Visitor<'a> {
    pub first_block: Option<BlockRef<'a, 'a>>,
    pub block: BlockRef<'a, 'a>,
    pub last_block: BlockRef<'a, 'a>,
    pub module: &'a Module<'a>,
    pub blocks: HashMap<NodeIndex, BlockRef<'a, 'a>>,
    pub blocks_args: HashMap<(NodeIndex, NodeIndex), Vec<cube::Value>>,
    pub current_region: RegionRef<'a, 'a>,
    pub context: &'a Context,
    pub location: Location<'a>,

    pub str_counter: usize,

    pub(self) values: Values<'a>,
    pub(self) args_manager: ArgsManager<'a>,
    pub liveness: Rc<MemoryLiveness>,
    pub mutable_variables: Vec<Id>,
    pub(self) stack_saves: HashMap<Id, StackSave<'a>>,
    pub(self) stack_save_counter: usize,
    pub(self) current_node: NodeIndex,
    pub(self) needs_parallelism: &'a mut bool,
}

#[derive(Clone, Copy)]
pub struct StackSave<'a> {
    pub stack_pointer: Value<'a, 'a>,
    pub seq: usize,
    pub alloc_block: NodeIndex,
}

impl<'a> Visitor<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        current_block: BlockRef<'a, 'a>,
        last_block: BlockRef<'a, 'a>,
        module: &'a Module<'a>,
        current_region: RegionRef<'a, 'a>,
        context: &'a Context,
        location: Location<'a>,
        args_manager: ArgsManager<'a>,
        liveness: Rc<MemoryLiveness>,
        needs_parallelism: &'a mut bool,
    ) -> Self {
        let blocks = HashMap::new();
        let blocks_args = HashMap::new();
        let str_counter = 0;
        let mut values = Values::new();
        let mutable_variables = Vec::new();

        for (&id, &buffer) in args_manager.buffers.iter() {
            values.insert(id, buffer);
        }

        Self {
            first_block: None,
            block: current_block,
            last_block,
            module,
            blocks,
            blocks_args,
            current_region,
            context,
            location,
            str_counter,
            args_manager,
            values,
            liveness,
            mutable_variables,
            stack_saves: HashMap::new(),
            stack_save_counter: 0,
            current_node: NodeIndex::new(0),
            needs_parallelism,
        }
    }

    pub fn get_block_args(
        &mut self,
        block_id: NodeIndex,
        destination: NodeIndex,
    ) -> Vec<Value<'a, 'a>> {
        let current_block = self.block;
        self.block = self.blocks[&block_id];
        let args = self
            .blocks_args
            .get(&(block_id, destination))
            .unwrap_or(&vec![])
            .iter()
            .map(|v| self.get_value(*v))
            .collect();
        self.block = current_block;
        args
    }

    pub fn append_global_str(&mut self, str_to_append: &str) -> String {
        let key = "str".to_string() + &self.str_counter.to_string();
        let str_value = StringAttribute::new(self.context, str_to_append).into();
        let str_type = llvm::r#type::array(
            IntegerType::new(self.context, 8).into(),
            str_to_append.len() as u32,
        );
        self.str_counter += 1;
        self.module.body().append_operation(llvm_ods::mlir_global(
            self.context,
            {
                let region = Region::new();
                let block = Block::new(&[]);
                let constant = block
                    .append_op_result(llvm_ods::mlir_constant(
                        self.context,
                        str_type,
                        str_value,
                        self.location,
                    ))
                    .unwrap();
                block.append_operation(llvm::r#return(Some(constant), self.location));
                region.append_block(block);
                region
            },
            TypeAttribute::new(str_type),
            StringAttribute::new(self.context, &key),
            linkage(self.context, Linkage::Internal),
            self.location,
        ));
        key
    }

    pub fn append_operation_with_result(
        &self,
        operation: impl Into<Operation<'a>>,
    ) -> Value<'a, 'a> {
        self.block
            .append_operation(operation)
            .result(0)
            .unwrap()
            .into()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn visit_kernel<'b: 'a>(
        context: &'a Context,
        location: Location<'a>,
        kernel: &'b KernelDefinition,
        module: &tracel_llvm::mlir_rs::ir::Module<'a>,
        func: &mut Function,
        global_state: &GlobalState,
        shared_memories: &SharedMemories,
        addr_type: StorageType,
        needs_parallelism: &mut bool,
    ) {
        let name = StringAttribute::new(context, "kernel");

        let attributes = &[(
            Identifier::new(context, "llvm.emit_c_interface"),
            Attribute::unit(context),
        )];

        let args = ArgsManagerBuilder::new(kernel, context, location, shared_memories, addr_type);

        let func_type = TypeAttribute::new(args.get_fn_type(context).into());

        add_external_function_to_module(context, module);
        add_sync_cube_function(context, module).unwrap();
        let liveness = func.analysis::<MemoryLiveness>(global_state);
        module.body().append_operation(func::func(
            context,
            name,
            func_type,
            {
                let region = Region::new();
                let args = args.create_top_block(&region, context, location);
                let block = region.first_block().unwrap();

                Self::insert_builtin_loop(
                    block,
                    module,
                    func,
                    context,
                    location,
                    args,
                    liveness,
                    needs_parallelism,
                )
                .unwrap();

                block.append_operation(func::r#return(&[], location));

                region
            },
            attributes,
            location,
        ));
    }

    #[allow(clippy::too_many_arguments)]
    pub(self) fn insert_builtin_loop(
        block: BlockRef<'a, 'a>,
        module: &tracel_llvm::mlir_rs::ir::Module<'a>,
        func: &Function,
        context: &'a Context,
        location: Location<'a>,
        mut args: ArgsManager<'a>,
        liveness: Rc<MemoryLiveness>,
        needs_parallelism: &mut bool,
    ) -> Result<(), Error> {
        let basic_block_id = func.root;
        let integer_type = IntegerType::new(context, 32).into();
        let start = block.const_int_from_type(context, location, 0, integer_type)?;
        let step = block.const_int_from_type(context, location, 1, integer_type)?;

        args.compute_derived_args_builtin(block, location);

        let cube_count_dim_x = block.muli(
            args.get(Builtin::CubeCountX),
            args.get(Builtin::CubeDimX),
            location,
        )?;
        let cube_count_dim_y = block.muli(
            args.get(Builtin::CubeCountY),
            args.get(Builtin::CubeDimY),
            location,
        )?;
        let cube_count_dim_x_usize = args.as_address_type(cube_count_dim_x, &block, location);
        let cube_count_dim_y_usize = args.as_address_type(cube_count_dim_y, &block, location);
        let cube_count_dim_xy_usize =
            block.muli(cube_count_dim_x_usize, cube_count_dim_y_usize, location)?;

        let cube_count_x_usize =
            args.as_address_type(args.get(Builtin::CubeCountX), &block, location);
        let cube_count_y_usize =
            args.as_address_type(args.get(Builtin::CubeCountY), &block, location);
        let cube_count_xy_usize = block.muli(cube_count_x_usize, cube_count_y_usize, location)?;

        block.append_operation(scf::r#for(
            start,
            args.get(Builtin::CubeCountZ),
            step,
            {
                let region = Region::new();
                let block = Block::new(&[(integer_type, location)]);
                args.set(Builtin::CubePosZ, block.argument(0)?.into());

                let absolute_pos_tmp_z = block.muli(
                    args.get(Builtin::CubePosZ),
                    args.get(Builtin::CubeDimZ),
                    location,
                )?;
                let absolute_pos_z =
                    block.addi(absolute_pos_tmp_z, args.get(Builtin::UnitPosZ), location)?;
                args.set(Builtin::AbsolutePosZ, absolute_pos_z);

                let absolute_pos_z_usize = args.as_address_type(absolute_pos_z, &block, location);
                let absolute_pos_z_corrected_usize =
                    block.muli(absolute_pos_z_usize, cube_count_dim_xy_usize, location)?;

                let cube_pos_z_usize =
                    args.as_address_type(args.get(Builtin::CubePosZ), &block, location);
                let cube_pos_z_corrected_usize =
                    block.muli(cube_pos_z_usize, cube_count_xy_usize, location)?;

                block.append_operation(scf::r#for(
                    start,
                    args.get(Builtin::CubeCountY),
                    step,
                    {
                        let region = Region::new();
                        let block = Block::new(&[(integer_type, location)]);
                        args.set(Builtin::CubePosY, block.argument(0)?.into());

                        let absolute_pos_tmp_y = block.muli(
                            args.get(Builtin::CubePosY),
                            args.get(Builtin::CubeDimY),
                            location,
                        )?;
                        let absolute_pos_y = block.addi(
                            absolute_pos_tmp_y,
                            args.get(Builtin::UnitPosY),
                            location,
                        )?;
                        args.set(Builtin::AbsolutePosY, absolute_pos_y);

                        let absolute_pos_y_usize =
                            args.as_address_type(absolute_pos_y, &block, location);
                        let absolute_pos_y_corrected_usize =
                            block.muli(absolute_pos_y_usize, cube_count_dim_x_usize, location)?;

                        let absolute_pos_xy_corrected_usize = block.addi(
                            absolute_pos_z_corrected_usize,
                            absolute_pos_y_corrected_usize,
                            location,
                        )?;

                        let cube_pos_y_usize =
                            args.as_address_type(args.get(Builtin::CubePosY), &block, location);
                        let cube_count_x_usize =
                            args.as_address_type(args.get(Builtin::CubeCountX), &block, location);
                        let cube_pos_y_corrected_usize =
                            block.muli(cube_pos_y_usize, cube_count_x_usize, location)?;
                        let cube_pos_yz_corrected_usize = block.addi(
                            cube_pos_z_corrected_usize,
                            cube_pos_y_corrected_usize,
                            location,
                        )?;

                        block.append_operation(scf::r#for(
                            start,
                            args.get(Builtin::CubeCountX),
                            step,
                            {
                                let region = Region::new();
                                let block = Block::new(&[(integer_type, location)]);
                                args.set(Builtin::CubePosX, block.argument(0)?.into());

                                let absolute_pos_tmp_x = block.muli(
                                    args.get(Builtin::CubePosX),
                                    args.get(Builtin::CubeDimX),
                                    location,
                                )?;
                                let absolute_pos_x = block.addi(
                                    absolute_pos_tmp_x,
                                    args.get(Builtin::UnitPosX),
                                    location,
                                )?;
                                args.set(Builtin::AbsolutePosX, absolute_pos_x);

                                let absolute_pos_x_usize =
                                    args.as_address_type(absolute_pos_x, &block, location);
                                let absolute_pos_usize = block.addi(
                                    absolute_pos_xy_corrected_usize,
                                    absolute_pos_x_usize,
                                    location,
                                )?;
                                args.set(Builtin::AbsolutePos, absolute_pos_usize);

                                let cube_pos_x_usize = args.as_address_type(
                                    args.get(Builtin::CubePosX),
                                    &block,
                                    location,
                                );
                                let cube_pos_usize = block.addi(
                                    cube_pos_yz_corrected_usize,
                                    cube_pos_x_usize,
                                    location,
                                )?;
                                args.set(Builtin::CubePos, cube_pos_usize);

                                region.append_block(block);
                                let current_block = region.first_block().unwrap();
                                let ops = current_block.append_operation(scf::execute_region(
                                    &[],
                                    Region::new(),
                                    location,
                                ));
                                let current_region = ops.region(0)?;

                                let last_block = Block::new(&[]);
                                last_block.append_operation(scf::r#yield(&[], location));
                                let last_block = current_region.append_block(last_block);

                                let mut visitor = Visitor::new(
                                    current_block,
                                    last_block,
                                    module,
                                    current_region,
                                    context,
                                    location,
                                    args,
                                    liveness.clone(),
                                    needs_parallelism,
                                );
                                visitor.visit_basic_block(basic_block_id, func);

                                current_block.append_operation(scf::r#yield(&[], location));
                                region
                            },
                            location,
                        ));

                        block.append_operation(scf::r#yield(&[], location));
                        region.append_block(block);
                        region
                    },
                    location,
                ));

                block.append_operation(scf::r#yield(&[], location));
                region.append_block(block);
                region
            },
            location,
        ));
        Ok(())
    }
}
