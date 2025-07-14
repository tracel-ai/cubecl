pub(super) mod args;
pub(super) mod block;
pub(super) mod elem;
pub(super) mod instruction;
pub(super) mod item;
pub(super) mod operation;
pub(super) mod prelude;
pub(super) mod variable;

use std::collections::HashMap;

use args::ArgsManager;
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
        r#type::{FunctionType, IntegerType, MemRefType},
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
    pub global_buffers: Vec<Value<'a, 'a>>,
    pub global_scalars: Vec<Value<'a, 'a>>,
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
        global_buffers: Vec<Value<'a, 'a>>,
        global_scalars: Vec<Value<'a, 'a>>,
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
            global_buffers,
            global_scalars,
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

        let mut inputs = Vec::with_capacity(kernel.buffers.len() + 9);
        let mut block_input = Vec::with_capacity(kernel.buffers.len());

        let mut global_buffers = vec![];
        let mut global_scalars = vec![];

        for binding in kernel.buffers.iter() {
            let inner_type = binding.item.elem.to_type(context);
            let memref = MemRefType::new(inner_type, &[i64::MIN], None, None).into();
            inputs.push(memref);
            block_input.push((memref, location));
        }

        for binding in kernel.scalars.iter() {
            let inner_type = binding.elem.to_type(context);
            let scalar = if binding.count > 1 {
                Type::vector(&[binding.count as u64], inner_type)
            } else {
                inner_type
            };
            inputs.push(scalar);
            block_input.push((scalar, location));
        }

        let integer_type: Type<'_> = IntegerType::new(context, 32).into();
        for _ in 0..9 {
            inputs.push(integer_type);
            block_input.push((integer_type, location));
        }

        let func_type = TypeAttribute::new(FunctionType::new(context, &inputs, &[]).into());

        add_external_function_to_module(context, module);
        module.body().append_operation(func::func(
            context,
            name,
            func_type,
            {
                let region = Region::new();
                let block = Block::new(&block_input);
                region.append_block(block);

                let block = region.first_block().unwrap();
                let argument_count = kernel.buffers.len();
                for i in 0..argument_count {
                    global_buffers.push(block.argument(i).unwrap().into());
                }

                let scalar = kernel.scalars.len();
                for i in argument_count..argument_count + scalar {
                    global_scalars.push(block.argument(i).unwrap().into());
                }

                Self::insert_builtin_loop(
                    block,
                    module,
                    opt,
                    context,
                    location,
                    global_buffers,
                    global_scalars,
                )
                .unwrap();

                block.append_operation(func::r#return(&[], location));

                region
            },
            attributes,
            location,
        ));
    }

    // TODO: cleanup this abomination by refactoring melior to make it at least not as bulky and verbose
    pub fn insert_builtin_loop(
        block: BlockRef<'a, 'a>,
        module: &tracel_llvm::melior::ir::Module<'a>,
        opt: &Optimizer,
        context: &'a Context,
        location: Location<'a>,
        global_buffers: Vec<Value<'a, 'a>>,
        global_scalars: Vec<Value<'a, 'a>>,
    ) -> Result<(), Error> {
        let basic_block_id = opt.entry();
        let integer_type = IntegerType::new(context, 32).into();
        let start = block.const_int_from_type(context, location, 0, integer_type)?;
        let step = block.const_int_from_type(context, location, 1, integer_type)?;

        let mut args = ArgsManager::default();

        // TODO refactor this in a macro
        let cube_dim_x = block
            .argument(global_buffers.len() + global_scalars.len())?
            .into();
        args.set_builtin(Builtin::CubeDimX, cube_dim_x);
        let cube_dim_y = block
            .argument(global_buffers.len() + global_scalars.len() + 1)?
            .into();
        args.set_builtin(Builtin::CubeDimY, cube_dim_y);
        let cube_dim_z = block
            .argument(global_buffers.len() + global_scalars.len() + 2)?
            .into();
        args.set_builtin(Builtin::CubeDimZ, cube_dim_z);
        let cube_count_x = block
            .argument(global_buffers.len() + global_scalars.len() + 3)?
            .into();
        args.set_builtin(Builtin::CubeCountX, cube_count_x);
        let cube_count_y = block
            .argument(global_buffers.len() + global_scalars.len() + 4)?
            .into();
        args.set_builtin(Builtin::CubeCountY, cube_count_y);
        let cube_count_z = block
            .argument(global_buffers.len() + global_scalars.len() + 5)?
            .into();
        args.set_builtin(Builtin::CubeCountZ, cube_count_z);
        let unit_pos_x = block
            .argument(global_buffers.len() + global_scalars.len() + 6)?
            .into();
        args.set_builtin(Builtin::UnitPosX, unit_pos_x);
        let unit_pos_y = block
            .argument(global_buffers.len() + global_scalars.len() + 7)?
            .into();
        args.set_builtin(Builtin::UnitPosY, unit_pos_y);
        let unit_pos_z = block
            .argument(global_buffers.len() + global_scalars.len() + 8)?
            .into();
        args.set_builtin(Builtin::UnitPosZ, unit_pos_z);

        let absolute_pos_tmp_0 = block.muli(cube_count_x, cube_dim_x, location)?;

        let absolute_pos_tmp_1 = block.muli(cube_count_y, cube_dim_y, location)?;

        let absolute_pos_tmp_2 = block.muli(absolute_pos_tmp_0, absolute_pos_tmp_1, location)?;

        let unit_pos_tmp0 = block.muli(cube_dim_y, cube_dim_z, location)?;

        let unit_pos_tmp1 = block.muli(unit_pos_tmp0, unit_pos_z, location)?;

        let unit_pos_tmp2 = block.muli(cube_dim_y, unit_pos_z, location)?;

        let unit_pos_tmp3 = block.muli(unit_pos_tmp1, unit_pos_tmp2, location)?;

        let unit_pos = block.muli(unit_pos_tmp3, unit_pos_x, location)?;
        args.set_builtin(Builtin::UnitPos, unit_pos);

        block.append_operation(scf::r#for(
            start,
            cube_count_x,
            step,
            {
                let region = Region::new();
                let block = Block::new(&[(integer_type, location)]);
                let cube_pos_x = block.argument(0)?.into();
                args.set_builtin(Builtin::CubePosX, cube_pos_x);

                let absolute_pos_x_tmp = block.muli(cube_pos_x, cube_dim_x, location)?;
                let absolute_pos_x = block.addi(absolute_pos_x_tmp, unit_pos_x, location)?;
                args.set_builtin(Builtin::AbsolutePosX, absolute_pos_x);

                block.append_operation(scf::r#for(
                    start,
                    cube_count_y,
                    step,
                    {
                        let region = Region::new();
                        let block = Block::new(&[(integer_type, location)]);
                        let cube_pos_y = block.argument(0)?.into();
                        args.set_builtin(Builtin::CubePosY, cube_pos_y);

                        let absolute_pos_y_tmp = block.muli(cube_pos_y, cube_dim_y, location)?;
                        let absolute_pos_y =
                            block.addi(absolute_pos_y_tmp, unit_pos_y, location)?;
                        args.set_builtin(Builtin::AbsolutePosY, absolute_pos_y);

                        let absolute_pos_tmp_3 =
                            block.muli(absolute_pos_y, absolute_pos_tmp_0, location)?;
                        block.append_operation(scf::r#for(
                            start,
                            cube_count_y,
                            step,
                            {
                                let region = Region::new();
                                let block = Block::new(&[(integer_type, location)]);

                                let cube_pos_z = block.argument(0)?.into();
                                args.set_builtin(Builtin::CubePosZ, cube_pos_z);

                                let absolute_pos_z_tmp =
                                    block.muli(cube_pos_z, cube_dim_z, location)?;
                                let absolute_pos_z =
                                    block.addi(absolute_pos_z_tmp, unit_pos_z, location)?;
                                args.set_builtin(Builtin::AbsolutePosZ, absolute_pos_z);

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
                                    global_buffers,
                                    global_scalars,
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
