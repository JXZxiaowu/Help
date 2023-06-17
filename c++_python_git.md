# C++
## 编译与链接
如 Java，当我们导入一个 module 时，我们不仅能够得到 declarations for imported symbols，并表明要链接他们。但是在C/C++ 中我们必须分开做这件事:
1. 通过 ·#include .h· 获得 declarations for imported symbols，这样编译器知道他们是什么
2. 链接 object file, static library and dynamic library 以访问代码
### object file 是什么
> An object file is the real output from the compilation phase. It's mostly machine code, but has info that allows a linker to see what symbols are in it as well as symbols it requires in order to work.   
A linker takes all these object files and combines them to form one executable.

object file 形式上包含:
- 可重定向代码
- 导出的符号/ exported symbols : 可以被其他编译单元使用的函数、类、变量等
- 导入的符号/ imported symbols : 本单元所使用的他其他编译单元的符号

### LIB and DLL(shared library)
- LIB : 静态库(.a or .lib)
- DLL : 动态库(.so)

将重复使用的代码/功能放入库中，linker 再将编译后的代码插入到程序中。

动态库则更进一步，动态库存储在本地，并不插入程序中，只在需要时被加载入内存。这也意味着在每一台机器上都必须有动态库副本。

### library and Object file
Static libraries are archives of object files。连接一个静态库和链接所有的 object files 是相同的。 

动态库则完全不同，它被看作是特殊的执行文件。不同于普通执行文件生成 entry point，动态库只声明自身的 imported and exported symbols. 在运行时，系统调用能够获得这些符号的地址并正常的使用
## Macro
### include 守卫
```
#ifndef TOKEN
#define TOKEN
/* code */
# else
/* code */
# endif
```
当头文件被 include，检查 TOKEN 是否被唯一定义，如果没有定义，就将会定义 TOKEN 并继续下面的代码。当文件被重复 include，执行 else 中的代码。它用来防止符号的多次声明。
## GNU
### gcc / g++ compiler
四个过程:
- preprocessing
- compilation proper
- assembly(汇编)
- libking

overall options   
`-c` : Preprocess, compile, and assemble into object files, but don't link   
`-S` : stop after compilation proper, do not assembly. 输出是汇编代码文件   
`-E` : stop after proprocessing, 输出是预处理后的源文件，将被放到标准输出里    
`-o file` : 将输出定向到文件 file. 无论是 preprocessed code, object file, assembler file or executable file   
`-fPIC` : 生成适用于 shared library 的 position-independent code. Such code accesee all constant addresses through a global offset table.   
`-Ldir` : Add directory dir to the list of directories to be searched for -l
`-llibrary/ -l library` : Search the library named library when linking. 如果同时存在 static and dynamic library, 除非 -static 被使用否则链接动态库.
`-shared` : 通过 object files 生成 shared library

生成静态库并连接
```
# Create the object files for the static library
gcc -c       src/tq84/add.c    -o bin/static/add.o
gcc -c       src/tq84/answer.c -o bin/static/answer.o

# Create static library
ar ar rcs bin/static/libtq84.a bin/static/add.o bin/static/answer.o

# Link statically
gcc  bin/main.o  -Lbin/static -ltq84 -o bin/statically-linked
```

生成动态库并连接
```
# object files for shared libraries need to be compiled as position independent
gcc -c -fPIC src/tq84/add.c    -o bin/shared/add.o
gcc -c -fPIC src/tq84/answer.c -o bin/shared/answer.o

# Create the shared library
gcc -shared bin/shared/add.o bin/shared/answer.o -o bin/shared/libtq84.so

# Link dynamically with the shared library
gcc  bin/main.o -Lbin/shared -ltq84 -o bin/use-shared-library

# configure the LD_LIBRARY_PATH if the shared library is not installed in a default location
LD_LIBRARY_PATH=$(pwd)
# execute
./use-shared-library
```
### ld linker
`ld -o main main.o func.o`
## Make and makefile
Make 以 makefile 为指导编译工程. 事实上 Make 使用 gcc/g++ 或其他编译器进行工作.   
当工程越来越大，所需要的编译指令也变得越多，所以将其放入脚本中，后来便形成了 built system (Make, Ninja)
### Makefile 语法
```
target : prerequisite
    command
    command
```
Make 决定运行 target 下的命令当 target file 不存在或者 prerequisite file 被修改.
### The all targets
执行 `make` 指令会运行第一个 target. 执行所有的 targets
```
all : one two

one: 
    touch one
two:
    touch two
```
### Multiple targets
```
all: f1.o f2.o

f1.o f2.o:
	echo $@
# Equivalent to:
# f1.o:
#	 echo f1.o
# f2.o:
#	 echo f2.o
```
### make vs make install
执行 make 时执行第一个 target; mask install 执行 install target.
## CMake
有了 Make 和 Makefile, 工程的比构建变得简单。但是对于不同的平台、编译器，Makefile 需要被重写才能完成构建工作。为了支持跨平台、跨编译器等，人们提出了 meta build system. CMake 便是其中之一，它通过 CMakefile.txt file、具体平台、编译器生成 Makefile 文件.

我们可以通过手写 Makefile 实现 build project, 但是 CMake 是跨平台的，当平台变化时，我们不需要重写 Makefile.
### Build and Run
```
cd Tutorial_build
cmake ../Tutorial       # 生成 build system
cmake --build .         # compile/link the project
```
> **Note:** `cmake ..` 如果出现 `nmake -？` 错误，使用 `cmake -S .. -G "MinGW Makefiles"` 代替.
### A basic starting
1. 开始于指定一个最低的 CMake 版本, 使用 `cmake_minimum_required(VERSION 3.10)`
2. 使用 project 命令指定工程名称, `project(Tutorial)`
3. 使用 add_executable() 指示 CMake 使用指定的源文件创建执行文件 `add_executable(Tutorial tutorial.cxx)`

> **add_executable()**: 使用指定的 source files 创建 executable. 使用 **add_source()**可以增加 source files.

### 指定 C++ 版本
4. 通过两个特殊变量 CMAKE_CXX_STANDARD 和 CMAKE_CXX_STANDARD_REQUIRED
```
# 确保 CMAKE_CXX_STANDARD 的 declaration 在 add_executable 的前面
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
```
### 指定工程版本和配置头文件
- `project(Tutorial VERSION 1.0)` 指定工程名称和版本   
> **Note:** 设置 cxx 变量 `set(CMAKE_CXX_COMPILER "D:/Soft/mingw64/bin/g++.exe"); set(CMAKE_C_COMPILER "D:/Soft/mingw64/bin/gcc.exe")` 在 project() 之前.
- `configure_file(TutorialConfig.h.in TutorialConfig.h)` 将带有 CMake 变量的 input file(.in) 拷贝到指定 include file, 同时替换其中的 CMake 变量. TutorialConfig.h.in 存在于 SOURCE_DIR 而 TutorialConfig.h 将会通过 TutorialConfig.h.in 生成到 BUILD_DIR 中
- `target_include_directory(Tutorial PUBLIC "${PROJECT_BINARY_DIR}")` add the binary tree to the search path for include files so that we will find TutorialConfig.h
- 创建定义了 CMake 变量的 input file, 变量的语法是`@VAR@`
```
# TutorialConfig.h.in

#define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
```

> **configure_file()**: .h.in -> .h file, include file 将会被放到 build 目录.
### 创建 Library
我们可以将将 project 组织成多个子目录的形式. 在每个子目录中存在多个 source files 以及一个 CMakeLists.txt. 在 top level 的 CMakeLists.txt 中使用 add_subdirectory 以构建子目录.

一旦 library 被创建，它通过 target_include_directories 和 target_link_libraries() 与可执行 target 连接起来.

例子:
```
Root
-- MathFunctions
    -- CMakeLists.txt
    -- MathFunctions.cxx
    -- MathFunctions.h
    -- mysqrt.cxx
    -- mysqrt.h
-- CMakeLists.txt
-- tutorial.cxx
```
1. 创建 library target
```
# MathFunctions/CMakeLists.txt

add_library(MathFunctions MathFunctions.cxx mysqrt.cxx)
```

> **add_library()**: 使用指定的 source files 创建 library.

2. 在 top level CMakeLists.txt 使用`add_subdirectory()`告知 build library target
```
# CMakeLists
add_subdirectory(MathFunctions)
```

> **add_subdirectory()**: Adds a subdirectory to the build

3. link library target to executable target
```
target_link_libraries(Tutorial PUBLIC MathFunctions)
```

> **target_link_libraries()**  指定需要链接的 library.
4. include files location
```
# CMakeLists.txt

target_include_directories(Tutorial PUBLIC 
                        "${PROJECT_BINARY_DIR}"
                        "${PROJECT_SOURCE_DIR/MathFunctions}")
# 一个为了寻找由 .in 生成的 TutorialConfig.h
# 一个为了寻找 library 的 include file
```

> **target_include_directories()** Specifies include directories to use when compiling a given target.
5. 在 tutorial.cxx 中使用该 library 的功能
```
# tutorial.cxx

#include "MathFunctions.h"
...
```
### 构建中的选项
让用户选择使用 custom implementation 或 standard implementation.

在本例中，让用户选择是否使用 MathFunctions 中的 mysqrt

1. 
    ```
    # MathFunctions/CMakeLists.txt

    option(USE_MYSQRT "Use custom sqrt root implementation" ON)     # 默认打开
    ```
2. when USE_MYSQRT is ON, 将 compile definitions USE_MYSQRT 放到 target 中
    ```
    # MathFunctions/CMakeLists.txt
    if (USE_MYSQRT)
        target_compile_definitions(MathFunctions PRIVATE "USE_MYSQRT")
    ```
    当 MY_SQRY is ON, USE_MYSQRT 将会被定义

> **target_compile_definitions()** Specifies compile definitions to use when compiling a given target
3. 根据 USE_MYSQRT 是否被定义更改 source file 中的代码
    ```
    # MathFunctions/MathFunctions.cxx
    ...
    #ifdef USE_MY_SQRT
        return detial::mysqrt(x)
    #else
        return std::sqrt(x)
    #endif
    ...
    ```
4. build and run `cmake ../Step2 -DUSE_MYSQRT=OFF, cmake --build .`
5. 如果 USE_MYSQRT 为 OFF, 那么 mysqrt.cxx 依然会被编译, 可以在 if block 中使用 add_sources() 来加入 mysqrt.cxx, 并将 add_library 改为 `add_library(MathFunctions MathFunctions.cxx)`

    另一种是根据 USE_MYSQRT将 mysqrt 单独编译成 library
    ```
    # MathFunctions/CMakeLists.txt
    if (USE_MYSQRT)
        add_library(SqrtLibrary STATIC mysqrt.cxx)
        target_link_library(MathFunctions PUBLIC Sqrt Library)
    ...
    #endif
    add_library(MathFunctions MathFunctions.cxx)
    ...
    ```
### Usage requirements for a library
使用 CMake 的方法定义 library usage requirements, 然后将 usage requirements 自动传递给 target.

1. We want to state that anybody linking to MathFunctions library needs to include the current source directory, while MathFunctions itself doesn't. This can be expressed with an INTERFACE usage requirement.
    ```
    # MathFunctions/CMakeLists
    target_include_directories(MathFunctions
                            INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    ```

> **CMAKE_CURRENT_SOURCE_DIR** : 指的是当前处理的 CMakeLists.txt 所在的路径   
> **PROJECT_SOURCE_DIR** : 工程顶层目录
> **PROJECT_BINARY_DIR** : Full path to build directory for project.

2. 我们能够安全地从 top level CmakeLists 中删除 `target_include_directories(Tutorial PUBLIC ${PROJECT_BINARY_DIR})`
### 使用 interface library 设置 C++ standard

### install
有时我们不仅需要 build 一个可执行 project, 也需要 installable. 使用 CMake install() 命令来表明安装规则. install 用于将 include files, library 拷贝到指定的位置.

1. install MathFunctions library and the headers file to lib and include 目录分别地.
    ```
    # MathFunctions/CMakeList
    set(installabel_libs MathFunctions tutorial_compiler_flags)
    if(TARGET SqrtLibrary)
        list(APPEND installabel_libs SqrtLibrary)
    endif()
    install(TARGET ${installable_libs} DESTINATION lib)
    install(FILES MathFunctions.h DESTINATION include)
    ```
    > **install(TARGET target DESTINATION)** 将 target 安装到指定目录
    
    > **set(variable value...)** set variable and its vakue.
    ```
    # CMakeList.txt
    install(TARGETS Tutorial DESTINATION bin)
    install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h" DESTINATION include)
    ```
    ### try compile
    测试一些功能/函数是否存在，并以此执行不同的动作.
    1. 我们将会使用 CheckCXXSourceCompiles 中的功能，因此
        ```
        # MathFunctions/CmakeList.txt
        include(CheckCXXSourceCompiles)
        ```
    2. Test of available functions of log and exp using check_cxx_compiles_source
        ```
        # MathFunctions/CmakeList.txt
        check_cxx_compiles_source(
            "
            #inlcude <cmath>
            int main(){
                std::log(1.0);
                return 0;
            }
            " HAVE_LOG
        )
        check_cxx_compiles_source(
            "
            #include <cmath>
            int main(){
                std::exp(1.0);
                return 0;
            }
            " HAVE_EXP
        )
        ```
> **check_cxx_compiles_source**: 该函数会尝试编译指定代码以测试特定的功能是否在平台存在.
3. 指定 compile definition
    ```
    ...
    if (HAVE_LOG AND HAVE_EXP)
        target_compile_definition(SqrtLibrary PRIVETE "HAVE_LOG" "HAVE_EXP")
    endif()
    ...
    ```
4. source fuke 根据 HAVE_LOG HAVE_EXP 执行不同的代码
    ```
    # MathFunctions/mysqrt.cxx
    #if defined(HAVE_LOG) && defined(HAVE_EXP)
        double result = std::exp(std::log(x) * 0.5);
        std::cout<< "..." <<std::endl;
    #else
        double result = x;
    ...
    ```
### custom command & generated file
MathFunctions/MakeTable.cxx 被用来生成 Table.h 以供 MathFUnctions/mysqrt.cxx 使用. 当 build project 时, 首先 build MakeTable executable 并执行 MakeTable 生成 Table.h, 之后编译 mysqrt.cxx 生成 MakeFunction library.
1. 创建 MathFunctions/MakeTable.cmake 并加入指令
    ```
    # MathFunctions/MakeTable.cmake
    add_executable(MakeTable MakeTable.cxx)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
        COMMOND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
        DEPENDS MakeTable
    )
    ```
2. 编辑 MathFunctions/CmakeList.txt
    ```
    add_library(SqrtLibrary STATIC mysqrt.cxx ${CAMKE_CURRENT_BINARY_DIR}/Table.h)
    target_inlcude_directories(SqrtLibrary PRIVATE ${CAME_CURRENT_BINARY_DIR})
    ```
4. include MakeTable.cmake at the top of the MathFunctions/CmakeList
    ```
    include(MakeTable.cmake)
    ```
### Packaging an installer
binary installation
```
# top-level CmakeList.txt

include(InstallRequiredSystembLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${Tutorial_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Tutorial_VERSION_MINOR}")
set(CPACK_SOURCE_GENERATOR "TGZ")
include(CPack)
```
> **InstallRequiredSystembLibraries** include any runtime libraries that are needed by the project for the current paltform.

我们正常 build project, 并运行 cpack executable 创建 binart distribution：
```
cpack --config CPackConfig.cmake
cpack --config CPackSourceConfig.cmake
```
### importing targets
IMPORTED 将当前 project 之外的 target 转换成逻辑上位于当前 project 的 target. 被 IMPORTED 的 target 不会产生 build 文件. 其分为两类
#### importing executable
假设现在我们已经 build & install a myexe executable. 现在我们将其导入到另外一个 CMake project. 首先使用 IMPORTED 告知 CMake 该 target 是一个引用自工程之外的 target, 之后使用 set_property() 告知该 target 的具体位置. 
```
add_executable(myexe IMPORTED)
set_property(TARGET myexe PROPERTY IMPORTED_LOCATION "../install_location/bin/myexe")
```
> **set_PROPERTY(TARGET )** reference set_target_properties()

最后我们可以在该工程中使用该 target.
#### importing libraries
其他项目的库也可以通过 IMPORTED 访问.
```
add_library(foo STATIC IMPORTED)
set_property(TARGET foo PROPERTY IMPORTED_LOCATION "/path_to_library/libfoo.a")
```
如果是 windows , 需要一起导入 .dll & .lib
```
add_library(foo STATIC IMPORTED)
set_property(TARGET foo PROPERTY IMPORTED_LOCATION "C:/PATH_TO_LIBRARY/foo.dll")
set_property(TARGET foo PROPERTY IMPORTED_LOCATION "C:/PATH_TO_LIBRARY/foo.lib")
```
使用库
```
add_executable(myexe src.cc)
target_link_libraries()myexe PRIVETE foo)
```
### Exportng Target
project 可以被设置生成必要的信息，以使他它能够被其他的项目所使用. 首先 build & install library.
```
#inlcude(GNUInstallDirs)
add_library(MahtFunction STATIC MathFunctions.cxx)

target_include_directories(MathFunctions PUBLIC
                            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
                            "$<INSTALL_NTERFACE:${CAMKE_INSTALL_INCLUDEDIR}>")
```
> **CAMKE_INSTALL_INCLUDEDIR**: GNUInstallDirs 定义的变量, 通常为一个相对于 install prefix 的路径. 

> 告知 CMake 在 build 和 install 时使用不同的 include 目录. 这是因为如果不这样做, 在创建 export information 时将输出一个特定于当前 build 目录的无效路径导致其他 project 无法使用.

install(TARGET) 用来安装 target, 指定其 property, 并绑定 export.
```
install(TARGET MathFunctions
        EXPORT MathFunctionsTargets
        LIBRARY DESTINATION ${CAMKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
```
安装 include files
```
install(FILES MathFunctions.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
```
> 在 install 的 DESTINATION 属性中不能使用与 build 有关的路径, 如 PROJECT_SOURCE_DIR, PROJECT_BINARY_DIR.

安装 export information, 实际上是生成 .camke 文件到指定目录
```
install(EXPORT MathFunctionsTargets 
        FILE MathFunctionsTargets.cmake 
        NAMESPACE MathFunctions:: 
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MathFunctions) 
```
MathFunctionsTargets.cmake 中的内容和 importing library 中手写的 IMPORT 类似, 但是有了 .cmake 后, 使用 include(.cmake) 命令即可引用 library.
```
include(${PREFIX}/${INSTALL_LIB}/cmake/Mathfunctions/MathfunctionsTargets.cmake)
...
target_link_libraries(myexe PRIVATE MathFunctions::Mathfunctions)
```
MathfunctionsTargets.cmake 文件负责 add_library 并给出库的位置及其他属性.

多个 target 可以绑定到一个 export name
```
# A/CMakeLists.txt
add_executable(myexe src1.c)
install(TARGETS myexe DESTINATION lib/myproj
        EXPORT myproj-targets)

# B/CMakeLists.txt
add_library(foo STATIC foo1.c)
install(TARGETS foo DESTINATION lib EXPORTS myproj-targets)

# Top CMakeLists.txt
add_subdirectory (A)
add_subdirectory (B)
install(EXPORT myproj-targets DESTINATION lib/myproj)
```
### Creating Package
当有了 export information 后, 我们通过 inlcude(.cmake file) 即可引用 library. 通过创建 Package 可以通过 find_package() 命令找到 project.

> **CMAKE_CURRENT_LIST_DIR** 当 CMake 处理你项目中的列表文件时，这个变量将始终被设置为当前正在处理的列表文件所在的目录.
1. 创建 package configuration file

    1. 书写 PackageNameConfig.cmake.in(用来生成PackageNameConfig.cmake)
        ```
        @PACKAGE_INIT@

        include("${CMAKE_CURRENT_LIST_DIR}/MathFunctionsTargets.cmake")

        check_required_components(MathFunctions)
        ```
    2. 一般境况下使用 configure_files() 从 .in file 生成对应的文件, 但是处理 package configuration file 时使用 configure_package_config_file()
    ```
    # CMakeList.txt
    configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/PackageNameConfig.cmake.in
                                "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfig.cmake"
                                INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/PackageName)
    ```
    3. install(FILES)
    ```
    #CMakeList.txt
    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/PackageNameConfig.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PackageName
            )
    ```
如果某一个 project 需要使用我们的 PackageName, 那么它仅需要
```
find_package(PackageName REQUIRED)
...
target_link_libraries(myexe PUBLIC PackageName)
```
2. 创建 package version file
当 find_package() 被调用时，CMake 读取该文件以确定是否与所需的版本兼容.
```
set(version 3.1.4)
# generate the version file for the config file

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfigVersion.cmake"
  VERSION "${version}"
  COMPATIBILITY AnyNewerVersion
)
```
COMPATIBILITY 为 AnyNewerVersion 意味着所安装的包的版本如果比要求的版本新或完全相同，将被视为兼容的.
### Exporting Targets from the build tree
通常情况下，一个 project 在被外部的 project 使用前经过了 build & install. 但是在一些情况下, 我们希望直至从 build tree 中 export targets. 然后这些 targets 能够在不经过 install 的情况下被其他 project 使用. 使用 export() 命令而不是 install(EXPORT) 来生成 export information.
```
#CMakeList.txt
export(EXPORT MathFunctions
        FILE “${CMAKE_CURRENT_BINARY_DIR/cmake/MathFunctionsTargets.cmake}”
        NAMESPACE MathFunctions::)
```
### Relocatable Packages
由 install(EXPORT) 创建的 package 是可重定位的. include directories 应该指定为相对于 CMAKE_INSTALL_PREFIX 的路径而且不能显示包含 CMAKE_INSTALL_PREFIX
```
target_include_directories(tgt INTERFACE $<INSTALL_INTERFACE:include>)

# wrong
target_include_directories(tgt INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

# OK, 使用 generator expression 作为占位符, 仅提示作用
target_include_directories(tgt INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)
```
### 使用 package configuration file
```
# CMakeList.txt
find_package(PackageName)
...
target_link_libraries(myexe PRIVATE packageName)
```
并且需要设置 CMAKE_PREFIX_PATH 变量, 有两种方式, path_to_install 只需到 lib 的上层即可.
1. set(CMAKE_PREFIX_PATH "path_to_install")
2. cmake -DCMAKE_PREFIX_PATH="path_to_install"


# C++ language
## 智能指针
### unqiue_ptr
```
// 手动实现：万能引用 + 完美转发
template <class T, class... Args>
unique_ptr<T> make_unique(Args&&... args){
  reutrn unique_ptr<T>(new T(std::forward(args)...));
}
```
- unique_ptr 堆内存进行唯一管理的行为，应该由程序员保证
- unique_prt 构造函数接收 T* 类型的指针，但是这样就失去了使用 unique_ptr 的作用了，如
  ```
  int *p = new int;
  unique_ptr<int> p1(p);
  // or
  unique_ptr<int> p1(new int(0));
  ```
- unique_ptr 删除了拷贝构造函数和拷贝辅助运算符，但是保留了移动拷贝赋值和移动拷贝构造函数，因此下列操作是可行的
  ```
  unique_ptr<int> p1 = make_unique<int>(0);
  unique_ptr<int> p2 = std::move(p1);
  ```
- release() 返回 T* 类型的指针，同时放弃管理权限
- reset() 回收内存，可以传入新的需要管理的内存

### shared_ptr
- make_shared<int>(0) 返回 shared_ptr 类型变量
- shared_ptr 存在拷贝构造函数和拷贝赋值运算符函数
- 没有 release 只有 reset()
### weak_ptr
- 无法操作资源，只用来查看资源是否被释放
- lock() 返回 shared_ptr 类型变量，用来进行资源的操作，如果已被释放则返回空指针
- use_count() 返回引用计数
- expired() 返回内存是否被释放

### auto_ptr C++17 delete
## 整数快速输入输出
```C++11
int read(){
    int x = 0, w = 1;
    char ch = 0
    while (ch<'0'||ch>'9'){
        if(ch == '-) w=-1;
        ch = getchar();
    }
    while(ch>='0'&&ch<'9'){
        x=(x<<1)+(x<<3)+(x^48);
        ch=getchar();
    }
    return x*w;
}
```
```C++11
int write(int num){
    if(num<0){
        num=-num;
        putchar('-');
    }
    static int stack[35];
    int top = 0;
    do{
        stack[top++] = num%10, num=num/10;
    } while(num);
    while(top) putchar(stack[--top] + 48);
}
```
# Git
git 有四种 space :
- workspace
- index file
- local
- remote

当我们修改一个文件时，改变是没有被缓存的(unstaged), 为了能够 commit, 我们必须 stage it—— add it to index using git add. 当我们执行 commit 时便是将加入 index file 中的东西提交.
## Git 练习
```
https://learngitbranching.js.org/?locale=zh_CN
```
## 开发场景
1. 初始化并和远程分支创建链接
  ```
  git init
  git remote add origin https://github.com/...
  git checkout -b master origin/master
  // 或者直接 clone
  git clone https://github.com/...
  ```
2. 拉取 feature 分支
  ```
  git checkout localFeature origin/remoteFeature
  ```
3. 在自己的本地 feature 分支上进行开发
  ```
  git add .
  git commit -m "new_message"
  ```
4.  push 到远程 feature 分支上

5. 在远程 feature 分支上执行 git rebase 合并到远程 master 分支上
## 本地常用命令
### checkout and HEAD
用来将 HEAD 指向指定节点/分支
- HEAD : 指向正在其基础上进行工作的提交记录，通常指向分支名。
- git checkout - b bugFix : 创建名为 bugFix 的分支并将 HEAD 切换过去
### branch
创建、修改、查看分支
- git branch bugFix HEAD : 在 HEAD 处创建新的分支
- git branch -a : 查看所有分支
- git branch -r : 查看所有远程分支
- git branch -f main HEAD : 移动分支 main 指向 HEAD
### merge and rebase
- merge 创建一个新的提交，该提交有两个父提交
- rebase base_branch change_branch : 会从两个分支的共同祖先开始提取待变分支上的修改，将待变分支指向基分支的最新提交，最后将提取的修改应用到基分支的最新提交的后面。
### reset and revert
- reset : 将当前分支回滚到指定分支
- revert : 创建新的提交，该提交会撤销指定提交的变更。
- 区别 : reset 无法共享，而 revert 创建新的提交，可以 push 到远程共享

<img src = "img/2023-05-30 172506.png">

### cherry-pick
cherry-pick 可以将提交树上任何地方的提交记录取过来追加到 HEAD 上（只要不是 HEAD 上游的提交就没问题）。
### describe
git describe HEAD : 查看描述
## 远程常用命令
### 远程分支
远程分支的的命名规范是 remote_name/branch_name   
当切换到远程分支或在远程分支上提交时，会出现 HEAD 与 origin/main 分离的情况，这是因为origin/main 只有在远程仓库中相应的分支更新了以后才会更新。
### set origin url
```
git remote set-url origin https://github.com/JXZxiaowu/Help.git
```
### fetch
git fetch 完成两步
- 从远程仓库下载本地仓库中缺失的提交记录
- 更新远程分支指针

git fetch 并不会改变你本地仓库的状态。它不会更新你的 main 分支，也不会修改你磁盘上的文件。所以, 你可以将 git fetch 的理解为单纯的下载操作 :D   
git fetch origin master : 只现在指定的远程分支并更新指针
### pull
使用 git fetch 获取远程数据后，我们可能想将远程仓库中的变化更新到我们的工作中。其实有很多方法：
- git cherry-pick origin/main
- git rebase origin/master
- git merge origin/master   
实际上，Git 提供了 git pull 完成这两个操作, git pull = git fetch; git merge :D
### push
git push 负责将你的变更上传到指定的远程仓库，并在远程仓库上合并你的新提交记录。
```
假设你周一克隆了一个仓库，然后开始研发某个新功能。到周五时，你新功能开发测试完毕，可以发布了。但是 —— 天啊！你的同事这周写了一堆代码，还改了许多你的功能中使用的 API，这些变动会导致你新开发的功能变得不可用。但是他们已经将那些提交推送到远程仓库了，因此你的工作就变成了基于项目旧版的代码，与远程仓库最新的代码不匹配了。

这种情况下, git push 就不知道该如何操作了。如果你执行 git push，Git 应该让远程仓库回到星期一那天的状态吗？还是直接在新代码的基础上添加你的代码，亦或由于你的提交已经过时而直接忽略你的提交？

因为这情况（历史偏离）有许多的不确定性，Git 是不会允许你 push 变更的。实际上它会强制你先合并远程最新的代码，然后才能分享你的工作。

origin: c0 -> c1 -> c2 <-main
local: c0 -> c1 -> C3 <-main
             ^
             |
             origin/main
我们想要提交 C3，但是 C3 基于远程分支中的 c1，而在远程仓库中分支已经更新到 c2 了，所以 Git 拒绝 git push :D
```
我们可以这样解决
- git fetch; git rebase origin/master; git push
- 尽管 git merge 创建新的分支，但是它告诉 Git 你已经合并了远程仓库的所有变更，也能够完成 push 操作
- git pull --rebase 相当于 git fetch 和 git rebase   
单参数的 push
- git push origin src : src 是一个本地分支
- git push origin src:dest : dest 是远程分支(如果不存在则创建)
### 跟踪远程分支
- git branch -u origin/main feature : 对于已存在的分支 feature 使其跟踪远程 main 分支
- git checkout -b feature origin/main : 创建分支 feature 并使其准总远程 main 分支
## 其他命令
- git log  查看提交记录
- git reflog  可查看修改记录，包括回退记录
- git reset --hard {commit id} 回退版本
- git stash 未被提交的代码放到暂存区
- git stash pop 还原并清除最近一次的暂存记录
- git remote -v 显示所有远程仓库
- git remote add url 添加一个远程仓库
- git remote rm name 删除远程仓库
- git commit --amend -m "new_message" 重命名最新一次的 commit 记录

# setuptools
setuptools 提供了一种方便的、标准化的方式构建和分享 Python 包，让开发者可以更加轻松地分享自己的代码，并帮助用户快速地安装和使用这些包.
## 基础的使用
```
# setup.py 最小配置
from setuptools import setup
setup(
    name = "mypackage",     
    version = "0.0.1",
    install_requires =[
        'requests',
        'importlib-metadata; python_version == "3.8"',
    ],
)
```
最后将代码组织成这种形式
```
mypackage
├── pyproject.toml  # and/or setup.cfg/setup.py (depending on the configuration method)
|   # README.rst or README.md (a nice description of your package)
|   # LICENCE (properly chosen license information, e.g. MIT, BSD-3, GPL-3, MPL-2, etc...)
└── mypackage
    ├── __init__.py
    └── ... (other Python files)
```
下面的命令会编译你的项目，并生成一个构建目录，其中包含了你的 Python 包的构建结果
```
python -m build
# or
python setup.py build
```
最后，将这个包安装到 Python 环境中
```
python install .
# or
pip isntall dist/PackageName.whl
```


## 发现 Package
### 指定 Package
最简单的情况是
```
setup(
    # ...
    packages = ["mypkg", "mypkg.subpkg1", "mypkg.subpkg2"]  # install .whl 后 import mypkg/mypkg.subpkg
)
```
这里注意如果我们仅有 mypkg, 那么 mypkg.subpkg 是不会被 build 的。
如果 mypkg 不在当前目录的根目录下，那么可以使用 package_dir 配置。
```
setup(
    # ...
    package_dir = {"": "src"}   # build 的对象将包含 src/mypkg, src/mypkg/subpkg
    # package_dir = {
    #     "mypkg": "src",
    #     "mypkg/subpkg": "src",
    # },
)
```
默认情况下 setup.py 路径下的所有包都会被包含。
### find_package
find_packages() takes a source directory and two lists of package name patterns to exclude and include, and then returns a list of str representing the packages it could find.
```
mypkg
├── pyproject.toml  # AND/OR setup.cfg, setup.py
└── src
    ├── pkg1
    │   └── __init__.py
    ├── pkg2
    │   └── __init__.py
    ├── additional
    │   └── __init__.py
    └── pkg
        └── namespace
            └── __init__.py
```
```
form setuptools import find_packages

setup(
    # ...
    packages = find_packages(
        where = 'src',  # '.' by default
        include = ['pkg*'],   # ['*']  by default   
    ),
    package_dir = {"":"src"},
    # ...
)
```
find_packages 返回 ['pkg1', 'pkg2', 'pkg']，所以必须使用 package_dir。如果是子模块，那么字符串为 mypkg.subpkg.
- 问题是在 src 下的没有 __init__.py 文件的目录不会被考虑（查找），这需要使用 find_namespace_packages 解决.
### find_namespace_packages
```
foo
├── pyproject.toml  # AND/OR setup.cfg, setup.py
└── src
    └── timmins
        └── foo
            └── __init__.py
```
需要使用
```
setup(
    # ...
    packages=find_namespace_packages(where='src'),
    package_dir={"": "src"}
    # ...
)
```
安装 .whl 后，import timmins.foo 使用.
### 检查 build 是否正确
在 build 你的文件之后，使用
```
tar tf dist/*.tar.gz
unzip -l dist/*.whl
```
查看是否有包没有被包含.
## 依赖管理
有三种依赖:
- build system requirement
- required dependency
- optional dependency
### build system requirement
如果是 setup.py，无需指定
### required dependency
```
setup(
    # ...
    install_requires=[
        'docutils',
        'BazSpam=1.1',
    ],
)
```
当 install project 时，所有未被安装的 dependency 都会被下载、build、安装。  
URL dependency
```
setup(
    # ...
    install_requires=[
        "Package-A @ git+https://example.net/package-a.git@main",
    ],
    # ...
)
```
### 最低 python 版本
```
setup(
    python_requires=">=3.6",
    # ...
)
```
## Build extension modules
setuptools can buld C/C++ extension modules. setup() 函数中的参数 ext_modules 接收 setuptools.Extension 对象列表. 例如
```
<project_folder>
├── pyproject.toml
└── foo.c
```
指示 setuptools 将 foo.c 编译成扩展模块 mylib.foo, 我们需要
```
from setuptools import setup, Extension

setup(
    ext_modules = [
        Extension(
            name="mylib.foo",   # as it would be imported
                                # may include packages/namespaces separated by '.'
            sources = ["foo.c"], # all sources are compiled into a single binary file
        ),
    ]
)
```
