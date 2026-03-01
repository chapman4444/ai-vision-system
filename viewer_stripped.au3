#Region
#AutoIt3Wrapper_Icon=C:\Ryan\Buttons\icons\Orbital_folder.ico
#AutoIt3Wrapper_Outfile_x64=viewer.exe
#AutoIt3Wrapper_Res_SaveSource=y
#AutoIt3Wrapper_Res_Language=1033
#AutoIt3Wrapper_Res_requestedExecutionLevel=highestAvailable
#AutoIt3Wrapper_Add_Constants=n
#AutoIt3Wrapper_Run_Au3Stripper=y
#Au3Stripper_Parameters=/pe + /mo
#EndRegion
Local $sScriptDir = @ScriptDir
Local $sExeName = StringTrimRight(@ScriptName, 4)
Local $sPythonScript = $sExeName & ".pyw"
If Not FileExists($sScriptDir & "\" & $sPythonScript) Then
MsgBox(16, "Error", "Python script '" & $sPythonScript & "' not found in directory '" & $sScriptDir & "'.")
Exit
EndIf
Local $sPythonExe = "pythonw.exe"
Local $sArgs = ""
For $i = 1 To $CmdLine[0]
$sArgs &= " " & '"' & $CmdLine[$i] & '"'
Next
Local $sCmd = $sPythonExe & ' "' & $sPythonScript & '"' & $sArgs
Local $pid = Run($sCmd, $sScriptDir, @SW_HIDE)
If @error Then
MsgBox(16, "Error", "Failed to run the command. Check if pythonw.exe is in PATH or accessible.")
EndIf
Exit
