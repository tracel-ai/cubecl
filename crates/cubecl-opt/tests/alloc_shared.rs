use cubecl_ir::prelude::*;
use cubecl_opt::passes::alloc_shared_memory::AllocateSharedMemoryBlockPass;
use pliron::{
    init_env_logger_for_tests,
    irfmt::parsers::spaced,
    operation::verify_operation,
    parsable::parse_from_str,
    pass::{AnalysisManager, Pass},
    result::ExpectOk,
};

/// Lengths of the shared arrays, in declaration order. Sizes differ so each
/// slice offset identifies one declaration.
const LENGTHS: [usize; 8] = [1, 3, 2, 5, 1, 7, 4, 6];
const ALIGN: usize = 4;
const ELEM: usize = 4;

fn expected_offsets() -> Vec<usize> {
    let mut cursor = 0;
    LENGTHS
        .iter()
        .map(|len| {
            let offset = cursor;
            cursor = (cursor + ELEM * len).next_multiple_of(ALIGN);
            offset
        })
        .collect()
}

fn kernel() -> String {
    let mut body = String::new();
    for (index, len) in LENGTHS.iter().enumerate() {
        let name = (b'a' + index as u8) as char;
        body.push_str(&format!(
            "        {name} = memory.declare_variable cube.array [cube.f32; {len}] Shared, align = {ALIGN}, init = cube.index 0;\n"
        ));
    }
    format!(
        "\
builtin.func @f: builtin.function <() -> ()> [] {{
  ^entry():
{body}    branch.return
}}
"
    )
}

/// `(ssa id, byte offset)` for each `slice_shared`, in block order.
fn slice_rows(printed: &str) -> Vec<(usize, usize)> {
    let mut rows = Vec::new();
    for line in printed.lines() {
        let Some(at) = line.find("slice_shared") else {
            continue;
        };
        let lhs = line[..at].split('=').next().unwrap_or("");
        let id_at = lhs
            .rfind("_v")
            .unwrap_or_else(|| panic!("slice result has no SSA id in: {line}"));
        let id: usize = lhs[id_at + 2..]
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("SSA id in: {line}"));
        let tail = &line[at..];
        let open = tail
            .find('[')
            .unwrap_or_else(|| panic!("slice offset missing in: {line}"));
        let close = tail[open..]
            .find(']')
            .unwrap_or_else(|| panic!("slice offset missing in: {line}"));
        let offset: usize = tail[open + 1..open + close]
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("offset in: {line}"));
        rows.push((id, offset));
    }
    rows
}

fn run_pass(ctx: &mut Context, input: &str) -> Result<String> {
    init_env_logger_for_tests!();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);
    verify_operation(op, ctx)?;
    AllocateSharedMemoryBlockPass.run(op, ctx, &mut AnalysisManager::default())?;
    verify_operation(op, ctx)?;
    Ok(Operation::get_op_dyn(op, ctx).disp(ctx).to_string())
}

#[test]
fn slice_shared_follows_declaration_order() -> Result<()> {
    let input = kernel();
    let expected = expected_offsets();
    // Each run builds a fresh HashMap. Its iteration order is not the order
    // the memories were declared, so a pass that walks `.values()` numbers
    // the slices differently every process. Declaration order does not.
    for _ in 0..24 {
        let ctx = &mut Context::default();
        let printed = run_pass(ctx, &input)?;
        let rows = slice_rows(&printed);
        assert_eq!(
            rows.len(),
            LENGTHS.len(),
            "expected one slice per declaration\n{printed}"
        );
        let offsets: Vec<usize> = rows.iter().map(|(_, offset)| *offset).collect();
        assert_eq!(
            offsets, expected,
            "offsets left declaration order\n{printed}"
        );
        let ids: Vec<usize> = rows.iter().map(|(id, _)| *id).collect();
        assert!(
            ids.windows(2).all(|pair| pair[0] < pair[1]),
            "slice SSA ids {ids:?} are not in declaration order\n{printed}"
        );
    }
    Ok(())
}
