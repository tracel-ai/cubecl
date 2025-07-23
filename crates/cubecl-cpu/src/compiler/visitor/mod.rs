pub(super) mod args_manager;
pub(super) mod block;
pub(super) mod elem;
pub(super) mod instruction;
pub(super) mod item;
pub(super) mod operation;
pub(super) mod prelude;
pub(super) mod variables;

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
        memref,
        ods::llvm as llvm_ods,
        scf,
    },
    ir::{
        Attribute, Block, BlockRef, Identifier, Location, Module, Operation, Region, RegionRef,
        attribute::{DenseElementsAttribute, StringAttribute, TypeAttribute},
        r#type::{IntegerType, MemRefType, RankedTensorType},
    },
};

use prelude::*;
use variables::Variables;

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

    pub str_counter: usize,

    pub(self) variables: Variables<'a>,
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
        opt: &Optimizer,
    ) -> Self {
        let blocks = HashMap::new();
        let blocks_args = HashMap::new();
        let str_counter = 0;
        let variables = Variables::new(opt);
        Self {
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
            variables,
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

        let mut args = ArgsManager::new(kernel, context, location);

        let func_type = TypeAttribute::new(args.get_fn_type(context).into());
        for const_array in opt.const_arrays() {
            let global = const_array.id;
            let name = global.to_string();
            let r#type = const_array.item.to_type(context);
            let memref = MemRefType::new(r#type, &[const_array.length as i64], None, None);
            let values: Vec<Attribute<'a>> = const_array
                .values
                .iter()
                .filter_map(|var| Visitor::into_attribute(context, *var, const_array.item))
                .collect();
            module.body().append_operation(memref::global(
                context,
                &name,
                None,
                memref,
                Some(
                    DenseElementsAttribute::new(
                        RankedTensorType::new(&[const_array.length as u64], r#type, None).into(),
                        &values,
                    )
                    .unwrap()
                    .into(),
                ),
                true,
                None,
                location,
            ));
        }
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
        let cube_count_dim_xy = block.muli(cube_count_dim_x, cube_count_dim_y, location)?;

        let cube_count_xy = block.muli(
            args.get(Builtin::CubeCountX),
            args.get(Builtin::CubeCountY),
            location,
        )?;

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

                let absolute_pos_z_corrected =
                    block.muli(absolute_pos_z, cube_count_dim_xy, location)?;

                let cube_pos_z_corrected =
                    block.muli(args.get(Builtin::CubePosZ), cube_count_xy, location)?;

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

                        let absolute_pos_y_corrected =
                            block.muli(absolute_pos_y, cube_count_dim_x, location)?;

                        let absolute_pos_xy_corrected = block.addi(
                            absolute_pos_z_corrected,
                            absolute_pos_y_corrected,
                            location,
                        )?;

                        let cube_pos_y_corrected = block.muli(
                            args.get(Builtin::CubePosY),
                            args.get(Builtin::CubeCountX),
                            location,
                        )?;
                        let cube_pos_yz_corrected =
                            block.addi(cube_pos_z_corrected, cube_pos_y_corrected, location)?;

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

                                let absolute_pos = block.addi(
                                    absolute_pos_xy_corrected,
                                    absolute_pos_x,
                                    location,
                                )?;
                                args.set(Builtin::AbsolutePos, absolute_pos);

                                let cube_pos = block.addi(
                                    cube_pos_yz_corrected,
                                    args.get(Builtin::CubePosX),
                                    location,
                                )?;
                                args.set(Builtin::CubePos, cube_pos);

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
                                    opt,
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
