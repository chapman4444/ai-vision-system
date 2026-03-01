#Region ;**** Directives created by AutoIt3Wrapper_GUI ****
#AutoIt3Wrapper_Icon=icons\Orbital_bow.ico
#AutoIt3Wrapper_Outfile_x64=llm_backup_service.exe
#AutoIt3Wrapper_Res_SaveSource=y
#AutoIt3Wrapper_Res_Language=1033
#AutoIt3Wrapper_Res_requestedExecutionLevel=highestAvailable
#AutoIt3Wrapper_Add_Constants=n
#EndRegion ;**** Directives created by AutoIt3Wrapper_GUI ****

Local $sScriptDir = @ScriptDir
Local $sExeName = StringTrimRight(@ScriptName, 4)
Local $sPythonScript = $sExeName & ".pyw"

If Not FileExists($sScriptDir & "\" & $sPythonScript) Then
    MsgBox(16, "Error", "Python script '" & $sPythonScript & "' not found in directory '" & $sScriptDir & "'.")
    Exit
EndIf

; Use pythonw.exe for .pyw files (no console window)
Local $sPythonExe = "pythonw.exe"
Local $sArgs = ""
For $i = 1 To $CmdLine[0]
    $sArgs &= " " & '"' & $CmdLine[$i] & '"'
Next

; Run directly without command shell for clean execution
Local $sCmd = $sPythonExe & ' "' & $sPythonScript & '"' & $sArgs

;MsgBox(0, "Command", "Running command: " & $sCmd)

; Run directly without @ComSpec - completely hidden
Local $pid = Run($sCmd, $sScriptDir, @SW_HIDE)
If @error Then
    MsgBox(16, "Error", "Failed to run the command. Check if pythonw.exe is in PATH or accessible.")
EndIf
Exit