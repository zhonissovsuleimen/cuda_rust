{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    cuda-oxide = {
      url = "github:NVlabs/cuda-oxide";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, cuda-oxide, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f (import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          }));
    in {
      devShells = forAllSystems (pkgs:
        let
          x11Libs = with pkgs; [
            libx11
            libxrandr
            libxcursor
            libxi
            libxinerama
            libGL
          ];
        in {
          default = pkgs.mkShell {
            inputsFrom = [
              cuda-oxide.devShells.${pkgs.system}.default
            ];

            packages = x11Libs;

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath x11Libs + ":/run/opengl-driver/lib";
          };
        });
    };
}
