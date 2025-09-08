#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_to_client {
    () => {
        use super::*;

        use cubecl_core::as_bytes;

        #[test]
        fn test_to_client() {
            let client_0 = TestRuntime::client(&CudaDevice { index: 0 });
            let client_1 = TestRuntime::client(&CudaDevice { index: 0 });

            let expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
            let input = client_0.create(f32::as_bytes(&expected));

            let output = client_0.to_client(input, &client_1).handle;

            let actual = client_1.read_one(output);
            let actual = f32::from_bytes(&actual);

            assert_eq!(actual, expected);
        }
    };
}
