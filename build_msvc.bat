@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "G:\cc workspace\rt"
if not exist build mkdir build
cl.exe /EHsc /std:c++17 /O2 /I include /I external /Fe:build\rt64.exe src\main.cc
