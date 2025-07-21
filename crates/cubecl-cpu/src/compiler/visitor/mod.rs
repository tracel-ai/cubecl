pub(super) mod args_manager;
pub(super) mod block;
pub(super) mod elem;
pub(super) mod instruction;
pub(super) mod item;
pub(super) mod operation;
pub(super) mod prelude;
pub(super) mod variable;

use std::collections::HashMap;

use args_manager::ArgsManager;
use cubecl_core::{ir::Builtin, prelude::KernelDefinition};
use cubecl_opt::{NodeIndex, Optimizer};
use tracel_llvm::melior::{
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
        attribute::{StringAttribute, TypeAttribute},
        r#type::IntegerType,
    },
};

use prelude::*;

use super::external_function::add_external_function_to_module;

pub struct Visitor<'a> {
    pub block: BlockRef<'a, 'a>,
    pub last_block: BlockRef<'a, 'a>,
    pub module: &'a Module<'a>,
    pub blocks: HashMap<NodeIndex, BlockRef<'a, 'a>>,
    pub blocks_args: HashMap<(NodeIndex, NodeIndex), Vec<Variable>>,
    pub current_region: RegionRef<'a, 'a>,
    pub context: &'a Context,
    pub location: Location<'a>,
    pub current_local_variables: HashMap<u32, Value<'a, 'a>>,
    pub current_version_variables: HashMap<(u32, u16), Value<'a, 'a>>,
    pub current_mut_variables: HashMap<u32, Value<'a, 'a>>,
    pub str_counter: usize,

    pub(self) args_manager: ArgsManager<'a>,
}

impl<'a> Visitor<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(self) fn new(
        current_block: BlockRef<'a, 'a>,
        last_block: BlockRef<'a, 'a>,
        module: &'a Module<'a>,
        current_region: RegionRef<'a, 'a>,
        context: &'a Context,
        location: Location<'a>,
        args_manager: ArgsManager<'a>,
    ) -> Self {
        let current_local_variables = HashMap::new();
        let current_version_variables = HashMap::new();
        let current_mut_variables = HashMap::new();
        let blocks = HashMap::new();
        let blocks_args = HashMap::new();
        let str_counter = 0;
        Self {
            block: current_block,
            last_block,
            module,
            blocks,
            blocks_args,
            current_region,
            context,
            location,
            current_local_variables,
            current_version_variables,
            current_mut_variables,
            str_counter,
            args_manager,
        }
    }

    pub fn get_block_args(
        &self,
        block_id: NodeIndex,
        destination: NodeIndex,
    ) -> Vec<Value<'a, 'a>> {
        self.blocks_args
            .get(&(block_id, destination))
            .unwrap_or(&vec![])
            .iter()
            .map(|v| self.get_variable(*v))
            .collect()
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

    pub(super) fn visit_kernel<'b: 'a>(
        context: &'a Context,
        location: Location<'a>,
        kernel: &'b KernelDefinition,
        module: &tracel_llvm::melior::ir::Module<'a>,
        opt: &Optimizer,
    ) {
        let name = StringAttribute::new(context, "kernel");

        let attributes = &[(
            Identifier::new(context, "llvm.emit_c_interface"),
            Attribute::unit(context),
        )];

        let mut args = ArgsManager::new(&kernel, context, location);

        let func_type = TypeAttribute::new(args.get_fn_type(context).into());

        add_external_function_to_module(context, module);
        module.body().append_operation(func::func(
            context,
            name,
            func_type,
            {
                let region = Region::new();
                region.append_block(args.create_top_block());
                let block = region.first_block().unwrap();

                Self::insert_builtin_loop(block, module, opt, context, location, args).unwrap();

                block.append_operation(func::r#return(&[], location));

                region
            },
            attributes,
            location,
        ));
    }

    pub(self) fn insert_builtin_loop(
        block: BlockRef<'a, 'a>,
        module: &tracel_llvm::melior::ir::Module<'a>,
        opt: &Optimizer,
        context: &'a Context,
        location: Location<'a>,
        mut args: ArgsManager<'a>,
    ) -> Result<(), Error> {
        let basic_block_id = opt.entry();
        let integer_type = IntegerType::new(context, 32).into();
        let start = block.const_int_from_type(context, location, 0, integer_type)?;
        let step = block.const_int_from_type(context, location, 1, integer_type)?;

        let absolute_pos_tmp_0 = block.muli(
            args.get_builtin(Builtin::CubeCountX),
            args.get_builtin(Builtin::CubeDimX),
            location,
        )?;

        let absolute_pos_tmp_1 = block.muli(
            args.get_builtin(Builtin::CubeCountY),
            args.get_builtin(Builtin::CubeDimY),
            location,
        )?;

        let absolute_pos_tmp_2 = block.muli(absolute_pos_tmp_0, absolute_pos_tmp_1, location)?;

        let unit_pos_tmp0 = block.muli(
            args.get_builtin(Builtin::CubeDimX),
            args.get_builtin(Builtin::CubeDimY),
            location,
        )?;

        let unit_pos_tmp1 =
            block.muli(unit_pos_tmp0, args.get_builtin(Builtin::UnitPosZ), location)?;

        let unit_pos_tmp2 = block.muli(
            args.get_builtin(Builtin::CubeDimX),
            args.get_builtin(Builtin::UnitPosY),
            location,
        )?;

        let unit_pos_tmp3 = block.addi(unit_pos_tmp1, unit_pos_tmp2, location)?;

        let unit_pos = block.addi(unit_pos_tmp3, args.get_builtin(Builtin::UnitPosX), location)?;
        args.set_builtin(Builtin::UnitPos, unit_pos);

        block.append_operation(scf::r#for(
            start,
            args.get_builtin(Builtin::CubeCountX),
            step,
            {
                let region = Region::new();
                let block = Block::new(&[(integer_type, location)]);
                let cube_pos_x = block.argument(0)?.into();
                args.set_builtin(Builtin::CubePosX, cube_pos_x);

                let absolute_pos_x_tmp =
                    block.muli(cube_pos_x, args.get_builtin(Builtin::CubeDimX), location)?;
                let absolute_pos_x = block.addi(
                    absolute_pos_x_tmp,
                    args.get_builtin(Builtin::UnitPosX),
                    location,
                )?;
                args.set_builtin(Builtin::AbsolutePosX, absolute_pos_x);

                block.append_operation(scf::r#for(
                    start,
                    args.get_builtin(Builtin::CubeCountY),
                    step,
                    {
                        let region = Region::new();
                        let block = Block::new(&[(integer_type, location)]);
                        let cube_pos_y = block.argument(0)?.into();
                        args.set_builtin(Builtin::CubePosY, cube_pos_y);

                        let absolute_pos_y_tmp = block.muli(
                            cube_pos_y,
                            args.get_builtin(Builtin::CubeDimY),
                            location,
                        )?;
                        let absolute_pos_y = block.addi(
                            absolute_pos_y_tmp,
                            args.get_builtin(Builtin::UnitPosY),
                            location,
                        )?;
                        args.set_builtin(Builtin::AbsolutePosY, absolute_pos_y);

                        let absolute_pos_tmp_3 =
                            block.muli(absolute_pos_y, absolute_pos_tmp_0, location)?;
                        block.append_operation(scf::r#for(
                            start,
                            args.get_builtin(Builtin::CubeCountY),
                            step,
                            {
                                let region = Region::new();
                                let block = Block::new(&[(integer_type, location)]);

                                let cube_pos_z = block.argument(0)?.into();
                                args.set_builtin(Builtin::CubePosZ, cube_pos_z);

                                let absolute_pos_z_tmp = block.muli(
                                    cube_pos_z,
                                    args.get_builtin(Builtin::CubeDimZ),
                                    location,
                                )?;
                                let absolute_pos_z = block.addi(
                                    absolute_pos_z_tmp,
                                    args.get_builtin(Builtin::UnitPosZ),
                                    location,
                                )?;
                                args.set_builtin(Builtin::AbsolutePosZ, absolute_pos_z);

                                let cube_pos_tmp = block.muli(
                                    args.get_builtin(Builtin::CubeDimX),
                                    args.get_builtin(Builtin::CubeDimY),
                                    location,
                                )?;
                                let cube_pos_tmp2 = block.muli(
                                    cube_pos_tmp,
                                    args.get_builtin(Builtin::CubePosZ),
                                    location,
                                )?;

                                let cube_pos_tmp3 = block.muli(
                                    args.get_builtin(Builtin::CubeDimX),
                                    args.get_builtin(Builtin::CubePosY),
                                    location,
                                )?;
                                let cube_pos_tmp4 =
                                    block.addi(cube_pos_tmp2, cube_pos_tmp3, location)?;
                                let cube_pos_tmp5 = block.addi(
                                    cube_pos_tmp4,
                                    args.get_builtin(Builtin::CubePosX),
                                    location,
                                )?;
                                args.set_builtin(Builtin::CubePos, cube_pos_tmp5);

                                let absolute_pos_tmp_4 =
                                    block.muli(absolute_pos_z, absolute_pos_tmp_2, location)?;
                                let absolute_pos_tmp_5 =
                                    block.addi(absolute_pos_x, absolute_pos_tmp_3, location)?;
                                let absolute_pos =
                                    block.addi(absolute_pos_tmp_5, absolute_pos_tmp_4, location)?;
                                args.set_builtin(Builtin::AbsolutePos, absolute_pos);

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
                                );
                                visitor.visit_basic_block(basic_block_id, opt);

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
