import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tzefa_Language.createdpython import *
print('VM TEST START')
set_current_line(1); add_var( 'INT', 'A', 75 ); tick_line() ;
set_current_line(2); add_var( 'INT', 'B', 50 ); tick_line() ;
set_current_line(3); add_var( 'INT', 'ZERO', 0 ); tick_line() ;
set_current_line(4); add_var( 'INT', 'RESULT', 0 ); tick_line() ;
set_current_line(5); vm_add_to( 'TEMPORARY', 'A', 'B' ); tick_line() ;
set_current_line(6); vm_print(get_var('INT','TEMPORARY'),True); tick_line() ;
set_current_line(7); vm_mod_to( 'TEMPORARY', 'A', 'B' ); tick_line() ;
set_current_line(8); vm_print(get_var('INT','TEMPORARY'),True); tick_line() ;
print_vars()
print('VM TEST END')
