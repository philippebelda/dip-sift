let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = with pkgs; [
    (python312.withPackages (python-pkgs: with python-pkgs; [
      numpy
      pillow
      matplotlib
      opencv-python
    ]))
  ];
}
