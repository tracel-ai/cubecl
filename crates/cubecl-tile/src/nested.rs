//! A walk of values that each hold more of their kind ([`each_and_nested`]).

/// Each of `values`, each followed by what `nested` gives of it, depth first.
pub(crate) fn each_and_nested<'a, T>(
    values: &'a [T],
    nested: fn(&'a T) -> Box<dyn Iterator<Item = &'a T> + 'a>,
) -> Box<dyn Iterator<Item = &'a T> + 'a> {
    Box::new(
        values
            .iter()
            .flat_map(move |value| core::iter::once(value).chain(nested(value))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Node {
        name: u32,
        children: Vec<Node>,
    }

    fn walk(node: &Node) -> Box<dyn Iterator<Item = &Node> + '_> {
        each_and_nested(&node.children, walk)
    }

    #[test]
    fn each_value_comes_before_what_it_holds() {
        let leaf = |name| Node {
            name,
            children: Vec::new(),
        };
        let values = [
            Node {
                name: 1,
                children: vec![leaf(2), leaf(3)],
            },
            leaf(4),
        ];
        let names: Vec<u32> = each_and_nested(&values, walk)
            .map(|node| node.name)
            .collect();
        assert_eq!(names, [1, 2, 3, 4]);
    }
}
