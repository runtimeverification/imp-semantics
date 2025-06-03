{
  lib,
  stdenv,
  makeWrapper,
  callPackage,

  k,
  which,

  kimp-pyk,
  rev ? null
} @ args:

stdenv.mkDerivation {
  pname = "kimp";
  version = if (rev != null) then rev else "dirty";
  buildInputs = [
    kimp-pyk
    k
  ];
  nativeBuildInputs = [ makeWrapper ];

  src = callPackage ../kimp-source { };

  dontUseCmakeConfigure = true;

  enableParallelBuilding = true;

  buildPhase = ''
    XDG_CACHE_HOME=$(pwd) ${
      lib.optionalString
      (stdenv.isAarch64 && stdenv.isDarwin)
      "APPLE_SILICON=true"
    } kimp-kdist -v build 'imp-semantics.*'
  '';

  installPhase = ''
    mkdir -p $out
    cp -r ./kdist-*/* $out/
    mkdir -p $out/bin
    makeWrapper ${kimp-pyk}/bin/kimp $out/bin/kimp --prefix PATH : ${lib.makeBinPath [ which k ]} --set KDIST_DIR $out
  '';
}