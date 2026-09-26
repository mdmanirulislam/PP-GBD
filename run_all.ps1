$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

foreach ($seed in 11, 23, 47, 59, 83) {
    python code/experiment.py models $seed
    python code/experiment.py dp $seed
    python code/experiment.py fed $seed
    python code/experiment.py ablations $seed
}
python code/experiment.py cost 11
python code/experiment.py agg
python code/figures_topologies.py
python code/figures_results.py
