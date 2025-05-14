{
  description = "A template for CubeCL";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, ... }@inputs: inputs.utils.lib.eachSystem [
    "x86_64-linux"
  ]
    (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "ssl-project";

          packages = with pkgs; [
            pkg-config
            openssl.dev
            openssl
            cmake
            ninja
            vulkan-loader
          ];
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.openssl pkgs.vulkan-loader ];
        };
      });
}
