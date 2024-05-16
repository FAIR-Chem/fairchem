packages_to_build="fairchem-core fairchem-data-oc fairchem-demo-ocpapi fairchem-applications-cattsunami"
pys="python3.10" # python3.9"

install_reqs_dev() {
    reqs="https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/packages/requirements.txt https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/packages/requirements-optional.txt"
    for req in $reqs; do
        pip install -r $req
    done
    pip install syrupy
}


install_reqs() {
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
    pip install syrupy
}

build_install_and_test_from_git_and_test() {( set -e
    py=$1
    install_reqs_dev

    git clone https://github.com/FAIR-Chem/fairchem.git
    pushd fairchem/packages
    for package in ${packages_to_build}; do
        pushd $package 
        pip install . 
        popd
    done
    popd

    $py -m pytest fairchem/tests
)}

build_install_and_test_from_pip_clone_git_and_test() {( set -e
    py=$1
    install_reqs

    for package in ${packages_to_build}; do
        pip install $package
    done
    
    git clone https://github.com/FAIR-Chem/fairchem.git
    pushd fairchem
    $py -m pytest fairchem/tests
    popd
)}

tmpdir=`mktemp -d` 
pushd $tmpdir

rm -f test_log

for py in $pys; do
    py_bn=`basename $py`
    mkdir -p $tmpdir/${py_bn}
    pushd $tmpdir/${py_bn}

    echo Building and testing $py in $tmpdir/${py_bn}
    ${py} -m venv ./venv_${py_bn}
    source ./venv_${py_bn}/bin/activate
    #build_install_and_test_from_git_and_test $py 2>&1 > ${py_bn}_log.txt
    build_install_and_test_from_pip_clone_git_and_test $py 2>&1 > ${py_bn}_log.txt
    echo $py $? >> test_log
    deactivate 

    popd
done
