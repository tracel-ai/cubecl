

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_to_client {
    () => {
        use super::*;

        #[test]
        fn test_to_client() {
            let client_0 = TestRuntime::client(&Default::default());
            let client_1 = TestRuntime::client(&Default::default());

            let expected = [0, 255, 0, 255];
            let handle = client_0.create_tensor(&expected, &[4], 1).handle;
            let handle = client_0.to_client(handle, &client_1).handle;

            let actual = client_1.read_one(handle);

            assert_eq!(&actual[0..4], expected);
        }
    };
}

