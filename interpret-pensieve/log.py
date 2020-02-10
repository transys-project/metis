# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Logging levels
NONE = 0
ERROR = 1
WARN = 2
INFO = 3
VERBOSE = 4
DEBUG = 5

# Current logging level
_cur_level = VERBOSE

# Current file logging level
_cur_flevel = VERBOSE

# Current log file (None for no file logging)
_cur_fname = None

# Sets the current logging level (for printing).
#
# level: int
def set_level(level):
    global _cur_level
    _cur_level = level

# Sets the current logging level (for log file).
#
# level: int
def set_file_level(level):
    global _cur_flevel
    _cur_flevel = level

# Sets the current log file.
#
# fname: str | None
def set_file(fname):
    global _cur_fname
    _cur_fname = fname

# Clears the current log file.
def clear_log():
    if not _cur_fname is None:
        f = open(_cur_fname, 'w')
        f.close()

# Logs the information
#
# s: str
def log(s, level):
    if level <= _cur_level:
        print(s)
    if level <= _cur_flevel and not _cur_fname is None:
        f = open(_cur_fname, 'a')
        f.write(s + '\n')
        f.close()
