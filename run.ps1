param(
    [ValidateSet('status','setup','download','prepare','test','probe','train','evaluate','audit','demo')]
    [string]$Task='status',
    [ValidateRange(1,100000)][int]$Steps=1000,
    [ValidateRange(1,32)][int]$BatchSize=8,
    [ValidateRange(1,50)][int]$Episodes=10,
    [string]$Checkpoint='checkpoints/single-task',
    [switch]$Resume
)
$ErrorActionPreference='Stop'
if($Task -eq 'demo') {
    $video=Join-Path $PSScriptRoot 'results/model-demo.mp4'
    if(-not (Test-Path -LiteralPath $video)) { throw 'Model video is not ready; run evaluate first.' }
    Invoke-Item -LiteralPath $video
    exit
}
$scriptPath=Join-Path $PSScriptRoot ('scripts\'+$(if($Task -eq 'setup') {'setup.sh'} else {'stage.sh'}))
$linuxPath='/mnt/'+$scriptPath.Substring(0,1).ToLower()+$scriptPath.Substring(2).Replace('\','/')
$arguments=@('-d','Ubuntu-24.04','--','bash',$linuxPath)
if($Task -ne 'setup') { $arguments += $Task }
if($Task -eq 'train') {
    $arguments += @('--steps',"$Steps",'--batch-size',"$BatchSize",'--output',$Checkpoint)
    if($Resume) { $arguments += '--resume' }
}
if($Task -eq 'evaluate') { $arguments += @('--episodes',"$Episodes",'--checkpoint',$Checkpoint) }
& wsl.exe @arguments
if($LASTEXITCODE -ne 0) { throw 'Stage failed. Inspect logs and the reported error.' }
