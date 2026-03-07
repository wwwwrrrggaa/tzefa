from Tzefa_Language.createdpython import *
set_current_line(1); add_var( 'INT', 'A', 75 ); tick_line() ;
set_current_line(2); add_var( 'INT', 'B', 50 ); tick_line() ;
set_current_line(3); add_var( 'INT', 'ZERO', 0 ); tick_line() ;
set_current_line(4); add_var( 'INT', 'R', 0 ); tick_line() ;
set_current_line(5); add_var( 'INT', 'RESULT', 0 ); tick_line() ;
set_current_line(6); add_var( 'STR', 'GCDIS', 'EMPTY' ); tick_line() ;
set_current_line(7); vm_print(get_var('INT','A'),False); tick_line() ;
set_current_line(8); vm_print(get_var('INT','B'),True); tick_line() ;
def COMPUTEGCD():
    set_current_line(10); add_local_cond( 'SZERO', 'EQUALS' ); tick_line() ;
    set_current_line(11); get_cond('SZERO').set_left(get_var('INT', 'B')); tick_line() ;
    set_current_line(12); get_cond('SZERO').set_right(get_var('INT', 'ZERO')); tick_line() ;
    if( set_current_line(13) and get_cond('SZERO').evaluate() and tick_line() ):
        set_current_line(14); return(exit_function_call('INT', 'A'))
    set_current_line(15); vm_mod_to( 'R', 'A', 'B' ); tick_line() ;
    set_current_line(16); vm_assign_int( 'A', 'B' ); tick_line() ;
    set_current_line(17); vm_assign_list( 'LOOPLIST', 'LOOPLIST' ); tick_line() ;
    enter_function_call('INT', 'A', COMPUTEGCD, 'INT', 'RESULT', 18)
    set_current_line(19); return(exit_function_call('INT', 'RESULT'))
enter_function_call('INT', 'A', COMPUTEGCD, 'INT', 'RESULT', 20)
set_current_line(21); vm_print(get_var('STR','GCDIS'),True); tick_line() ;
print_vars()