# prevent laptop from sleeping

$wsh = New-Object -ComObject WScript.Shell
while ($true) {
  $wsh.SendKeys('+{F15}')
  Start-Sleep -Seconds 59
}
