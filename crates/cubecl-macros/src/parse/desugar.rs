use std::mem::take;

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{
    Attribute, Expr, ExprMatch, Index, Local, LocalInit, Pat, PatIdent, PatSlice, PatStruct,
    PatTuple, PatTupleStruct, Stmt, parse_quote,
    spanned::Spanned,
    visit_mut::{self, VisitMut},
};

use crate::{
    generate::expression::inner_pat,
    parse::{expression::is_runtime_compatible_variant, helpers::is_comptime_attr},
};

/// Internal macro marking another copy of a guarded `match`'s scrutinee; parsed into
/// [`Expression::CloneUnchecked`](crate::expression::Expression::CloneUnchecked).
pub const MATCH_SCRUTINEE_MACRO: &str = "__cube_match_scrutinee";

pub struct Desugar;
impl VisitMut for Desugar {
    fn visit_block_mut(&mut self, i: &mut syn::Block) {
        let mut next_id = 0;
        let stmts = desugar_pats(take(&mut i.stmts), &mut next_id);

        i.stmts = stmts;
        visit_mut::visit_block_mut(self, i)
    }

    fn visit_expr_mut(&mut self, i: &mut Expr) {
        // Children first, so guarded matches nested in arms or guards are already rewritten.
        visit_mut::visit_expr_mut(self, i);

        if let Expr::Match(mat) = i
            && mat.arms.iter().any(|arm| matches!(arm.pat, Pat::Guard(_)))
        {
            let Expr::Match(mat) = std::mem::replace(i, Expr::Verbatim(TokenStream::new())) else {
                unreachable!()
            };
            *i = desugar_match_guards(mat);
        }
    }
}

/// One `match` arm with its guard split off the pattern.
struct GuardedArm {
    attrs: Vec<Attribute>,
    pat: Pat,
    guard: Option<Expr>,
    body: Expr,
}

/// Rewrites a `match` with guarded arms (`Pat if cond => body`) into guard-free matches, since
/// the cube expansion only understands plain patterns.
///
/// A guarded arm becomes `Pat => if cond { body } else { <fallthrough> }`, where the fallthrough
/// re-matches the scrutinee against the later arms that could also match `Pat`. The guard can then
/// be a comptime or runtime condition like any other `if`.
///
/// ```ignore
/// match mode {
///     Mode::A if x > 0 => 1,
///     Mode::A => 2,
///     Mode::B => 3,
/// }
/// // becomes
/// match copy_of!(mode) {
///     Mode::A => if x > 0 { 1 } else { match copy_of!(mode) { Mode::A => 2, Mode::B => 3 } },
///     Mode::B => 3,
/// }
/// ```
///
/// `copy_of!` stands for [`MATCH_SCRUTINEE_MACRO`]: each match takes its own copy of the scrutinee
/// so the original is never moved before a fallthrough re-matches it.
///
/// Each fallthrough match also gets the remaining unguarded arms appended so Rust still sees it as
/// exhaustive. Those arms never run, and are left out when an earlier arm already covers them.
fn desugar_match_guards(mat: ExprMatch) -> Expr {
    let ExprMatch {
        mut attrs,
        expr,
        arms,
        ..
    } = mat;

    let arms = arms
        .into_iter()
        .map(|arm| match arm.pat {
            Pat::Guard(guarded) => GuardedArm {
                attrs: arm.attrs,
                pat: *guarded.pat,
                guard: Some(*guarded.guard),
                body: *arm.body,
            },
            pat => GuardedArm {
                attrs: arm.attrs,
                pat,
                guard: None,
                body: *arm.body,
            },
        })
        .collect::<Vec<_>>();

    // The rewritten matches see fewer arms than the original, which can flip how the cube parser
    // classifies them. Keep the classification the original would get without its guards.
    if !is_runtime_match(&arms) && !attrs.iter().any(is_comptime_attr) {
        attrs.push(parse_quote!(#[comptime]));
    }

    // Re-matching needs the scrutinee more than once, so anything but a plain variable is
    // evaluated once up front.
    let (binding, scrutinee) = match *expr {
        Expr::Path(path) if path.qself.is_none() && path.path.get_ident().is_some() => {
            (None, Expr::Path(path))
        }
        expr => {
            let ident = format_ident!("__match_guard_scrutinee");
            let binding: Stmt = parse_quote!(let #ident = #expr;);
            (Some(binding), parse_quote!(#ident))
        }
    };

    let builder = MatchBuilder {
        attrs: &attrs,
        scrutinee: &scrutinee,
        arms: &arms,
    };
    let all = (0..arms.len()).collect::<Vec<_>>();
    let mat = builder.build(&all, &[]);

    match binding {
        Some(binding) => parse_quote!({ #binding #mat }),
        None => mat,
    }
}

struct MatchBuilder<'a> {
    attrs: &'a [Attribute],
    scrutinee: &'a Expr,
    arms: &'a [GuardedArm],
}

impl MatchBuilder<'_> {
    /// Builds a guard-free match that picks the first arm of `seq` (indices into `arms`, in order)
    /// whose pattern and guard both hold. `filler` arms are appended only to keep the match
    /// exhaustive; the caller guarantees they can't be reached.
    fn build(&self, seq: &[usize], filler: &[usize]) -> Expr {
        let mut emitted: Vec<&Pat> = Vec::new();
        let mut out: Vec<(&[Attribute], &Pat, TokenStream)> = Vec::new();

        for (pos, &idx) in seq.iter().enumerate() {
            let arm = &self.arms[idx];
            // An earlier arm already takes every value this one matches. If that arm was guarded,
            // this one is handled in its fallthrough.
            if emitted.iter().any(|prev| subsumes(prev, &arm.pat)) {
                continue;
            }

            let GuardedArm {
                attrs, pat, body, ..
            } = arm;
            match &arm.guard {
                Some(guard) => {
                    let later = seq[pos + 1..]
                        .iter()
                        .copied()
                        .filter(|&next| !disjoint(pat, &self.arms[next].pat))
                        .collect::<Vec<_>>();
                    let filler = (0..self.arms.len())
                        .filter(|i| self.arms[*i].guard.is_none() && !later.contains(i))
                        .collect::<Vec<_>>();
                    let fallthrough = self.build(&later, &filler);
                    out.push((
                        attrs,
                        pat,
                        quote! {{ if #guard { #body } else { #fallthrough } }},
                    ));
                }
                None => out.push((attrs, pat, body.to_token_stream())),
            }
            emitted.push(pat);
        }

        for &idx in filler {
            let GuardedArm {
                attrs, pat, body, ..
            } = &self.arms[idx];
            if emitted.iter().any(|prev| subsumes(prev, pat)) {
                continue;
            }
            out.push((attrs, pat, body.to_token_stream()));
            emitted.push(pat);
        }

        let scrutinee = self.scrutinee;
        let copy = format_ident!("{MATCH_SCRUTINEE_MACRO}");

        // A lone catch-all arm needs no match. Emitting one anyway would leave a match with no
        // cases, which the cube parser can't lower.
        if let [(_, pat, body)] = out.as_slice() {
            match unparen(pat) {
                Pat::Wild(_) => return parse_quote!({ #body }),
                Pat::Ident(ident) if is_binding(ident) => {
                    return parse_quote!({
                        let #ident = #copy!(#scrutinee);
                        #body
                    });
                }
                _ => {}
            }
        }

        let attrs = self.attrs;
        let arms = out
            .iter()
            .map(|(attrs, pat, body)| quote! { #(#attrs)* #pat => #body, });
        parse_quote! {
            #(#attrs)*
            match #copy!(#scrutinee) {
                #(#arms)*
            }
        }
    }
}

/// Mirrors the cube parser's check for lowering a match to a runtime switch on the discriminant.
fn is_runtime_match(arms: &[GuardedArm]) -> bool {
    let num_values = arms.iter().filter_map(|arm| inner_pat(&arm.pat)).count();
    !arms.is_empty()
        && num_values <= 1
        && arms
            .iter()
            .all(|arm| is_runtime_compatible_variant(&arm.pat))
        && !arms.iter().all(|arm| matches!(arm.pat, Pat::Wild(_)))
}

fn unparen(pat: &Pat) -> &Pat {
    match pat {
        Pat::Paren(pat) => unparen(&pat.pat),
        pat => pat,
    }
}

/// Whether an identifier pattern is a binding rather than a constant or unit variant like `None`.
fn is_binding(pat: &PatIdent) -> bool {
    pat.subpat.is_none()
        && pat
            .ident
            .to_string()
            .trim_start_matches('_')
            .starts_with(|c: char| c.is_lowercase())
}

/// Whether `pat` matches every value.
fn is_irrefutable(pat: &Pat) -> bool {
    match unparen(pat) {
        Pat::Wild(_) | Pat::Rest(_) => true,
        Pat::Ident(ident) => is_binding(ident),
        Pat::Tuple(tuple) => tuple.elems.iter().all(is_irrefutable),
        _ => false,
    }
}

/// The path of a variant, struct or constant pattern, with whether all its fields are irrefutable.
fn path_pattern(pat: &Pat) -> Option<(String, bool)> {
    let path = |path: &syn::Path| path.to_token_stream().to_string();
    match unparen(pat) {
        Pat::Path(pat) if pat.qself.is_none() => Some((path(&pat.path), true)),
        Pat::Ident(ident) if !is_binding(ident) && ident.subpat.is_none() => {
            Some((ident.ident.to_string(), true))
        }
        Pat::TupleStruct(pat) if pat.qself.is_none() => {
            Some((path(&pat.path), pat.elems.iter().all(is_irrefutable)))
        }
        Pat::Struct(pat) if pat.qself.is_none() => Some((
            path(&pat.path),
            pat.fields.iter().all(|field| is_irrefutable(&field.pat)),
        )),
        _ => None,
    }
}

/// Conservatively checks that every value matching `inner` also matches `outer`.
fn subsumes(outer: &Pat, inner: &Pat) -> bool {
    let (outer, inner) = (unparen(outer), unparen(inner));
    if is_irrefutable(outer) {
        return true;
    }
    if let Pat::Or(inner) = inner {
        return inner.cases.iter().all(|case| subsumes(outer, case));
    }
    if let Pat::Or(outer) = outer {
        return outer.cases.iter().any(|case| subsumes(case, inner));
    }
    if let (Pat::Lit(outer), Pat::Lit(inner)) = (outer, inner) {
        return outer.to_token_stream().to_string() == inner.to_token_stream().to_string();
    }
    match (path_pattern(outer), path_pattern(inner)) {
        (Some((outer_path, true)), Some((inner_path, _))) => {
            outer_path == inner_path && same_kind(outer, inner)
        }
        _ => false,
    }
}

fn same_kind(a: &Pat, b: &Pat) -> bool {
    std::mem::discriminant(unparen(a)) == std::mem::discriminant(unparen(b))
}

/// Conservatively checks that no value can match both patterns. Only distinct literals and
/// distinctly named variants count, so arms are never dropped from a fallthrough by mistake.
fn disjoint(a: &Pat, b: &Pat) -> bool {
    let (a, b) = (unparen(a), unparen(b));
    if let Pat::Or(a) = a {
        return a.cases.iter().all(|case| disjoint(case, b));
    }
    if let Pat::Or(b) = b {
        return b.cases.iter().all(|case| disjoint(a, case));
    }
    if let (Pat::Lit(a), Pat::Lit(b)) = (a, b) {
        let (a, b) = (
            a.to_token_stream().to_string(),
            b.to_token_stream().to_string(),
        );
        // Only plain integer, char or bool literals compare reliably by spelling.
        let simple = |s: &str| {
            s.chars().all(|c| c.is_ascii_digit())
                || s.starts_with('\'')
                || s == "true"
                || s == "false"
        };
        return simple(&a) && simple(&b) && a != b;
    }
    match (variant_name(a), variant_name(b)) {
        (Some(a), Some(b)) => a != b,
        _ => false,
    }
}

/// Last path segment of a pattern that names an enum variant. Constants (`SCREAMING_CASE`, or a
/// lone capital letter) are left out since two different constants can hold the same value.
fn variant_name(pat: &Pat) -> Option<String> {
    let path = match pat {
        Pat::Path(pat) => &pat.path,
        Pat::TupleStruct(pat) => &pat.path,
        Pat::Struct(pat) => &pat.path,
        Pat::Ident(ident) if !is_binding(ident) && ident.subpat.is_none() => {
            let name = ident.ident.to_string();
            return name.contains(char::is_lowercase).then_some(name);
        }
        _ => return None,
    };
    let name = path.segments.last()?.ident.to_string();
    // Tuple and struct patterns always name a type or variant, never a constant.
    let is_ctor = matches!(pat, Pat::TupleStruct(_) | Pat::Struct(_));
    (is_ctor || name.contains(char::is_lowercase)).then_some(name)
}

fn desugar_pats(stmts: Vec<Stmt>, next_id: &mut usize) -> Vec<Stmt> {
    let mut output = Vec::new();
    for stmt in stmts {
        match stmt {
            Stmt::Local(Local {
                pat: Pat::Struct(pat),
                init: Some(init),
                ..
            }) => {
                let stmts = desugar_struct_destructure(pat, init, *next_id);
                *next_id += 1;
                output.extend(desugar_pats(stmts, next_id));
            }
            Stmt::Local(Local {
                pat:
                    Pat::Tuple(PatTuple { elems, .. }) | Pat::TupleStruct(PatTupleStruct { elems, .. }),
                init: Some(init),
                ..
            }) => {
                let stmts = desugar_tuple_destructure(elems, init, *next_id);
                *next_id += 1;
                output.extend(desugar_pats(stmts, next_id));
            }
            Stmt::Local(Local {
                pat: Pat::Slice(PatSlice { elems, .. }),
                init: Some(init),
                ..
            }) => {
                let elems = elems.into_iter().collect::<Vec<_>>();
                let stmts = desugar_slice_destructure(&elems, init, *next_id);
                *next_id += 1;
                output.extend(desugar_pats(stmts, next_id));
            }
            stmt => output.push(stmt),
        }
    }
    output
}

fn desugar_struct_destructure(pat: PatStruct, init: LocalInit, id: usize) -> Vec<Stmt> {
    let init_ident = format_ident!("__struct_destructure_init_{id}");
    let fields = pat.fields.into_iter().map(|field| {
        let attrs = field.attrs;
        let pat = field.pat;
        let member = field.member;
        quote_spanned! {pat.span()=>
            #(#attrs)* let #pat = #init_ident.#member;
        }
    });
    let init = init.expr;
    let init = quote_spanned![init.span()=> let #init_ident = #init;];
    parse_quote! {
        #init
        #(#fields)*
    }
}

fn desugar_tuple_destructure(
    fields: impl IntoIterator<Item = Pat>,
    init: LocalInit,
    id: usize,
) -> Vec<Stmt> {
    let init_ident = format_ident!("__tuple_destructure_init_{id}");
    let fields = fields.into_iter().enumerate().map(|(i, pat)| {
        let member = Index::from(i);
        quote_spanned! {pat.span()=>
            let #pat = #init_ident.#member;
        }
    });
    let init = init.expr;
    let init = quote_spanned![init.span()=> let #init_ident = #init;];
    parse_quote! {
        #init
        #(#fields)*
    }
}

fn desugar_slice_destructure(fields: &[Pat], init: LocalInit, id: usize) -> Vec<Stmt> {
    if let Some(field) = fields.iter().find(|field| {
        matches!(
            field,
            Pat::Ident(PatIdent {
                subpat: Some(_),
                ..
            })
        )
    }) {
        let err = syn::Error::new(field.span(), "@ patterns are not currently supported")
            .to_compile_error();
        return vec![parse_quote!(#err;)];
    }

    // Slice patterns can't have more than one rest pattern, so it can always be cleanly separated
    // into before rest (which start at 0) and after rest (which start from len - n_after_rest).
    let rest_pos = fields
        .iter()
        .position(|field| matches!(field, Pat::Rest(_)))
        .unwrap_or(fields.len());
    let from_start = &fields[..rest_pos];
    let from_end = &fields[(rest_pos + 1).min(fields.len())..];
    let init_ident = format_ident!("__slice_destructure_init_{id}");
    let len_ident = format_ident!("__slice_destructure_len_{id}");

    let from_start_fields = from_start.iter().enumerate().map(|(i, pat)| {
        let offset = Index::from(i);
        quote_spanned! {pat.span()=>
            let #pat = #init_ident[#offset];
        }
    });

    let init = init.expr;

    let len_expr = if from_end.is_empty() {
        quote![]
    } else {
        quote_spanned![init.span()=> let #len_ident = #init_ident.len();]
    };
    let from_end_fields = from_end.iter().enumerate().map(|(i, pat)| {
        let offset = Index::from(from_end.len() - i);
        // This requires a bit of a hack on `sub::expand` to make it work for `Sequence`
        quote_spanned! {pat.span()=>
            let #pat = #init_ident[#len_ident - #offset];
        }
    });

    let init = quote_spanned![init.span()=> let #init_ident = #init;];
    parse_quote! {
        #init
        #len_expr
        #(#from_start_fields)*
        #(#from_end_fields)*
    }
}

#[cfg(test)]
mod tests {
    use super::Desugar;
    use crate::{expression::Block, scope::Context};
    use syn::{Ident, parse_quote, visit_mut::VisitMut};

    #[test]
    fn nested_tuple_patterns_are_fully_desugared() {
        let mut block: syn::Block = parse_quote!({
            let (a, b, (c, d, (e, f))) = tuple;
        });

        Desugar.visit_block_mut(&mut block);

        let mut context = Context::new(parse_quote!(()), false, false);
        let result = Block::from_block(block, &mut context);

        assert!(result.is_ok(), "{result:?}");

        let bindings: [Ident; 6] = [
            parse_quote!(a),
            parse_quote!(b),
            parse_quote!(c),
            parse_quote!(d),
            parse_quote!(e),
            parse_quote!(f),
        ];
        for binding in bindings {
            assert!(
                context.variable(&binding).is_some(),
                "missing binding `{binding}`"
            );
        }
    }

    // Commas after block-bodied arms are optional, and rustfmt drops them from the expected
    // side, so they're stripped before comparing.
    fn normalize(tokens: proc_macro2::TokenStream) -> String {
        tokens.to_string().replace("} ,", "}")
    }

    fn desugar_expr(expr: syn::Expr) -> String {
        let mut block: syn::Block = parse_quote!({ #expr });
        Desugar.visit_block_mut(&mut block);
        normalize(quote::quote!(#block))
    }

    fn expected(block: syn::Block) -> String {
        normalize(quote::quote!(#block))
    }

    #[test]
    fn guard_falls_through_to_same_variant() {
        let got = desugar_expr(parse_quote! {
            match var {
                Var::One if cond => a,
                Var::One => b,
                Var::Two => c,
            }
        });
        let want = expected(parse_quote!({
            match __cube_match_scrutinee!(var) {
                Var::One => {
                    if cond {
                        a
                    } else {
                        match __cube_match_scrutinee!(var) {
                            Var::One => b,
                            Var::Two => c,
                        }
                    }
                }
                Var::Two => c,
            }
        }));
        assert_eq!(got, want);
    }

    #[test]
    fn guard_falls_through_to_wildcard_body() {
        let got = desugar_expr(parse_quote! {
            match mode {
                Mode::Nan if x < lo || x > hi => x - lo,
                _ => offset,
            }
        });
        let want = expected(parse_quote!({
            match __cube_match_scrutinee!(mode) {
                Mode::Nan => {
                    if x < lo || x > hi {
                        x - lo
                    } else {
                        { offset }
                    }
                }
                _ => offset,
            }
        }));
        assert_eq!(got, want);
    }

    // Arms of other variants can't follow a failed guard, so they stay out of the fallthrough
    // apart from the unguarded ones needed for exhaustiveness.
    #[test]
    fn fallthrough_skips_disjoint_guarded_arms() {
        let got = desugar_expr(parse_quote! {
            match var {
                Var::One if a => 1,
                Var::Two if b => 2,
                Var::One => 3,
                Var::Two => 4,
            }
        });
        let want = expected(parse_quote!({
            match __cube_match_scrutinee!(var) {
                Var::One => {
                    if a {
                        1
                    } else {
                        match __cube_match_scrutinee!(var) {
                            Var::One => 3,
                            Var::Two => 4,
                        }
                    }
                }
                Var::Two => {
                    if b {
                        2
                    } else {
                        match __cube_match_scrutinee!(var) {
                            Var::Two => 4,
                            Var::One => 3,
                        }
                    }
                }
            }
        }));
        assert_eq!(got, want);
    }

    // Three payload arms keep the original off the runtime-switch path, so every rewritten match
    // is marked comptime to stay on it too.
    #[test]
    fn chained_guards_on_one_variant_stay_linear() {
        let got = desugar_expr(parse_quote! {
            match var {
                Var::One(x) if x > 0 => 1,
                Var::One(x) if x < -5 => 2,
                Var::One(_) => 3,
            }
        });
        let want = expected(parse_quote!({
            #[comptime]
            match __cube_match_scrutinee!(var) {
                Var::One(x) => {
                    if x > 0 {
                        1
                    } else {
                        #[comptime]
                        match __cube_match_scrutinee!(var) {
                            Var::One(x) => {
                                if x < -5 {
                                    2
                                } else {
                                    #[comptime]
                                    match __cube_match_scrutinee!(var) {
                                        Var::One(_) => 3,
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }));
        assert_eq!(got, want);
    }

    #[test]
    fn complex_scrutinee_is_evaluated_once() {
        let got = desugar_expr(parse_quote! {
            match self.mode() {
                Mode::A if c => 1,
                _ => 2,
            }
        });
        let want = expected(parse_quote!({
            {
                let __match_guard_scrutinee = self.mode();
                match __cube_match_scrutinee!(__match_guard_scrutinee) {
                    Mode::A => {
                        if c {
                            1
                        } else {
                            { 2 }
                        }
                    }
                    _ => 2,
                }
            }
        }));
        assert_eq!(got, want);
    }

    #[test]
    fn guarded_binding_arm_keeps_its_binding() {
        let got = desugar_expr(parse_quote! {
            match n {
                x if x > 3 => x,
                x => x + 1,
            }
        });
        let want = expected(parse_quote!({
            {
                let x = __cube_match_scrutinee!(n);
                {
                    if x > 3 {
                        x
                    } else {
                        {
                            let x = __cube_match_scrutinee!(n);
                            x + 1
                        }
                    }
                }
            }
        }));
        assert_eq!(got, want);
    }

    #[test]
    fn match_without_guards_is_untouched() {
        let got = desugar_expr(parse_quote! {
            match var {
                Var::One => 1,
                Var::Two => 2,
            }
        });
        let want = expected(parse_quote!({
            match var {
                Var::One => 1,
                Var::Two => 2,
            }
        }));
        assert_eq!(got, want);
    }

    #[test]
    fn guarded_match_parses_into_cube_expression() {
        let mut block: syn::Block = parse_quote!({
            match var {
                Var::One if cond => 1u32,
                Var::One => 2u32,
                Var::Two => 3u32,
            }
        });
        Desugar.visit_block_mut(&mut block);

        let mut context = Context::new(parse_quote!(u32), false, false);
        context.push_variable(parse_quote!(var), None, false, false);
        context.push_variable(parse_quote!(cond), None, false, false);
        let result = Block::from_block(block, &mut context);
        assert!(result.is_ok(), "{result:?}");
    }
}
