pub(super) mod block;
pub(super) mod elem;
pub(super) mod instruction;
pub(super) mod item;
pub(super) mod operation;
pub(super) mod prelude;
pub(super) mod variable;

use std::collections::HashMap;

use cubecl_core::prelude::KernelDefinition;
use cubecl_opt::{NodeIndex, Optimizer};
use tracel_llvm::melior::{
    Context,
    dialect::{arith, func, scf},
    ir::{
        Attribute, Block, BlockRef, Identifier, Location, Operation, Region, RegionRef,
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, MemRefType},
    },
};

use prelude::*;

use super::external_function::add_external_function_to_module;

pub struct Visitor<'a> {
    pub current_block: BlockRef<'a, 'a>,
    pub last_block: BlockRef<'a, 'a>,
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

    pub cube_dim_x: Value<'a, 'a>,
    pub cube_dim_y: Value<'a, 'a>,
    pub cube_dim_z: Value<'a, 'a>,
    pub cube_count_x: Value<'a, 'a>,
    pub cube_count_y: Value<'a, 'a>,
    pub cube_count_z: Value<'a, 'a>,
    pub unit_pos_x: Value<'a, 'a>,
    pub unit_pos_y: Value<'a, 'a>,
    pub unit_pos_z: Value<'a, 'a>,
    pub cube_pos_x: Value<'a, 'a>,
    pub cube_pos_y: Value<'a, 'a>,
    pub cube_pos_z: Value<'a, 'a>,
    pub absolute_pos_x: Value<'a, 'a>,
    pub absolute_pos_y: Value<'a, 'a>,
    pub absolute_pos_z: Value<'a, 'a>,
    pub absolute_pos: Value<'a, 'a>,
}

impl<'a> Visitor<'a> {
    pub fn new(
        current_block: BlockRef<'a, 'a>,
        last_block: BlockRef<'a, 'a>,
        current_region: RegionRef<'a, 'a>,
        context: &'a Context,
        location: Location<'a>,
        global_buffers: Vec<Value<'a, 'a>>,
        global_scalars: Vec<Value<'a, 'a>>,
        cube_dim_x: Value<'a, 'a>,
        cube_dim_y: Value<'a, 'a>,
        cube_dim_z: Value<'a, 'a>,
        cube_count_x: Value<'a, 'a>,
        cube_count_y: Value<'a, 'a>,
        cube_count_z: Value<'a, 'a>,
        unit_pos_x: Value<'a, 'a>,
        unit_pos_y: Value<'a, 'a>,
        unit_pos_z: Value<'a, 'a>,
        cube_pos_x: Value<'a, 'a>,
        cube_pos_y: Value<'a, 'a>,
        cube_pos_z: Value<'a, 'a>,
        absolute_pos_x: Value<'a, 'a>,
        absolute_pos_y: Value<'a, 'a>,
        absolute_pos_z: Value<'a, 'a>,
        absolute_pos: Value<'a, 'a>,
    ) -> Self {
        let current_local_variables = HashMap::new();
        let current_version_variables = HashMap::new();
        let current_mut_variables = HashMap::new();
        let blocks = HashMap::new();
        let blocks_args = HashMap::new();

        Self {
            current_block,
            last_block,
            blocks,
            blocks_args,
            current_region,
            context,
            location,
            current_local_variables,
            current_version_variables,
            current_mut_variables,
            global_buffers,
            global_scalars,
            cube_dim_x,
            cube_dim_y,
            cube_dim_z,
            cube_count_x,
            cube_count_y,
            cube_count_z,
            unit_pos_x,
            unit_pos_y,
            unit_pos_z,
            cube_pos_x,
            cube_pos_y,
            cube_pos_z,
            absolute_pos_x,
            absolute_pos_y,
            absolute_pos_z,
            absolute_pos,
        }
    }

    pub fn block(&self) -> BlockRef<'a, 'a> {
        self.current_block.clone()
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

    pub fn append_operation_with_result(
        &self,
        operation: impl Into<Operation<'a>>,
    ) -> Value<'a, 'a> {
        self.block()
            .append_operation(operation.into())
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
            Attribute::unit(context).into(),
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

        for _ in 0..9 {
            let index = Type::index(context);
            inputs.push(index);
            block_input.push((index, location));
        }

        let func_type = TypeAttribute::new(FunctionType::new(context, &inputs, &[]).into());

        let location = location;
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
                    opt,
                    context,
                    location,
                    global_buffers,
                    global_scalars,
                );

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
        opt: &Optimizer,
        context: &'a Context,
        location: Location<'a>,
        global_buffers: Vec<Value<'a, 'a>>,
        global_scalars: Vec<Value<'a, 'a>>,
    ) {
        let basic_block_id = opt.entry();
        let c0: Value<'_, '_> = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(Type::index(context), 0).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        let c1: Value<'_, '_> = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(Type::index(context), 1).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // TODO refactor this in a macro
        let cube_dim_x = block
            .argument(global_buffers.len() + global_scalars.len())
            .unwrap()
            .into();
        let cube_dim_y = block
            .argument(global_buffers.len() + global_scalars.len() + 1)
            .unwrap()
            .into();
        let cube_dim_z = block
            .argument(global_buffers.len() + global_scalars.len() + 2)
            .unwrap()
            .into();
        let cube_count_x = block
            .argument(global_buffers.len() + global_scalars.len() + 3)
            .unwrap()
            .into();
        let cube_count_y = block
            .argument(global_buffers.len() + global_scalars.len() + 4)
            .unwrap()
            .into();
        let cube_count_z = block
            .argument(global_buffers.len() + global_scalars.len() + 5)
            .unwrap()
            .into();
        let unit_pos_x = block
            .argument(global_buffers.len() + global_scalars.len() + 6)
            .unwrap()
            .into();
        let unit_pos_y = block
            .argument(global_buffers.len() + global_scalars.len() + 7)
            .unwrap()
            .into();
        let unit_pos_z = block
            .argument(global_buffers.len() + global_scalars.len() + 8)
            .unwrap()
            .into();

        let absolute_pos_tmp_0: Value<'a, 'a> = block
            .append_operation(arith::muli(cube_count_x, cube_dim_x, location))
            .result(0)
            .unwrap()
            .into();

        let absolute_pos_tmp_1: Value<'a, 'a> = block
            .append_operation(arith::muli(cube_count_y, cube_dim_y, location))
            .result(0)
            .unwrap()
            .into();

        let absolute_pos_tmp_2: Value<'a, 'a> = block
            .append_operation(arith::muli(
                absolute_pos_tmp_0,
                absolute_pos_tmp_1,
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        block.append_operation(
            scf::r#for(
                c0,
                cube_count_x,
                c1,
                {
                    let region = Region::new();
                    let block = Block::new(&[(Type::index(&context), location)]);
                    let cube_pos_x = block.argument(0).unwrap().into();

                    let absolute_pos_x_tmp = block
                        .append_operation(arith::muli(cube_pos_x, cube_dim_x, location))
                        .result(0)
                        .unwrap()
                        .into();
                    let absolute_pos_x = block
                        .append_operation(arith::addi(absolute_pos_x_tmp, unit_pos_x, location))
                        .result(0)
                        .unwrap()
                        .into();

                    block.append_operation(scf::r#for(
                        c0,
                        cube_count_y,
                        c1,
                        {
                            let region = Region::new();
                            let block = Block::new(&[(Type::index(&context), location)]);
                            let cube_pos_y = block.argument(0).unwrap().into();

                            let absolute_pos_y_tmp = block
                                .append_operation(arith::muli(cube_pos_y, cube_dim_y, location))
                                .result(0)
                                .unwrap()
                                .into();
                            let absolute_pos_y = block
                                .append_operation(arith::addi(
                                    absolute_pos_y_tmp,
                                    unit_pos_y,
                                    location,
                                ))
                                .result(0)
                                .unwrap()
                                .into();
                            let absolute_pos_tmp_3 = block
                                .append_operation(arith::muli(
                                    absolute_pos_y,
                                    absolute_pos_tmp_0,
                                    location,
                                ))
                                .result(0)
                                .unwrap()
                                .into();
                            block.append_operation(scf::r#for(
                                c0,
                                cube_count_y,
                                c1,
                                {
                                    let region = Region::new();
                                    let block = Block::new(&[(Type::index(&context), location)]);

                                    let cube_pos_z = block.argument(0).unwrap().into();

                                    let absolute_pos_z_tmp = block
                                        .append_operation(arith::muli(
                                            cube_pos_z, cube_dim_z, location,
                                        ))
                                        .result(0)
                                        .unwrap()
                                        .into();
                                    let absolute_pos_z = block
                                        .append_operation(arith::addi(
                                            absolute_pos_z_tmp,
                                            unit_pos_z,
                                            location,
                                        ))
                                        .result(0)
                                        .unwrap()
                                        .into();

                                    let absolute_pos_tmp_4 = block
                                        .append_operation(arith::muli(
                                            absolute_pos_z,
                                            absolute_pos_tmp_2,
                                            location,
                                        ))
                                        .result(0)
                                        .unwrap()
                                        .into();
                                    let absolute_pos_tmp_5 = block
                                        .append_operation(arith::addi(
                                            absolute_pos_x,
                                            absolute_pos_tmp_3,
                                            location,
                                        ))
                                        .result(0)
                                        .unwrap()
                                        .into();

                                    let absolute_pos = block
                                        .append_operation(arith::addi(
                                            absolute_pos_tmp_5,
                                            absolute_pos_tmp_4,
                                            location,
                                        ))
                                        .result(0)
                                        .unwrap()
                                        .into();

                                    region.append_block(block);
                                    let current_block = region.first_block().unwrap();
                                    let ops = current_block.append_operation(scf::execute_region(
                                        &[],
                                        Region::new(),
                                        location,
                                    ));
                                    let current_region = ops.region(0).unwrap();

                                    let last_block = Block::new(&[]);
                                    last_block.append_operation(scf::r#yield(&[], location).into());
                                    let last_block = current_region.append_block(last_block);

                                    let mut visitor = Visitor::new(
                                        current_block,
                                        last_block,
                                        current_region,
                                        context,
                                        location,
                                        global_buffers,
                                        global_scalars,
                                        cube_dim_x,
                                        cube_dim_y,
                                        cube_dim_z,
                                        cube_count_x,
                                        cube_count_y,
                                        cube_count_z,
                                        unit_pos_x,
                                        unit_pos_y,
                                        unit_pos_z,
                                        cube_pos_x,
                                        cube_pos_y,
                                        cube_pos_z,
                                        absolute_pos_x,
                                        absolute_pos_y,
                                        absolute_pos_z,
                                        absolute_pos,
                                    );
                                    visitor.visit_basic_block(basic_block_id, opt);

                                    current_block.append_operation(scf::r#yield(&[], location));
                                    region
                                },
                                location,
                            ));
                            block.append_operation(scf::r#yield(&[], location).into());
                            region.append_block(block);
                            region
                        },
                        location,
                    ));

                    block.append_operation(scf::r#yield(&[], location).into());
                    region.append_block(block);
                    region
                },
                location,
            )
            .into(),
        );
    }
}
