from createdpython import * 
line(1); addvar( "INT" , 'INTEGERONE' , 32 ); endline() ;
line(2); addvar( "INT" , 'INTEGERTWO' , 4 ); endline() ;
line(3); addvar( "LIST" , 'GCDLIST' , getvar( 'INT' , 'TWO' ).read() ); endline() ;
line(4); getvar('LIST','GCDLIST').changeindex(getvar( 'INT' , 'ZERO' ).read()); endline()
line(5);getvar('LIST','GCDLIST') .placevalue('INTEGERONE',"INT"); endline()
line(6); getvar('LIST','GCDLIST').changeindex(getvar( 'INT' , 'ONE' ).read()); endline()
line(7);getvar('LIST','GCDLIST') .placevalue('INTEGERTWO',"INT"); endline()
line(8); addcond( 'GCDCOMPARE' , 'BIGGER' ); endline() ;
line(9); getcond( 'GCDCOMPARE' ).changeleft(getvar( "INT" , 'INTEGERTWO' )); endline() ;
line(10); getcond( 'GCDCOMPARE' ).changeright(getvar( "INT" , 'ZERO' )); endline() ;
def GCD():
    line(12); getvar('LIST','GCDLIST').changeindex(getvar( 'INT' , 'ZERO' ).read()); endline()
    line(13);getvar('INT','INTEGERONE').copyvar(getvar('LIST','GCDLIST').read()); endline()
    line(14); getvar('LIST','GCDLIST').changeindex(getvar( 'INT' , 'ONE' ).read()); endline()
    line(15);getvar('INT','INTEGERTWO').copyvar(getvar('LIST','GCDLIST').read()); endline()
    line(16); mod( 'INTEGERONE' , 'INTEGERTWO' ); endline()
    line(17); assignint( 'INTEGERONE' , 'INTEGERTWO' ); endline()
    line(18); assignint( 'INTEGERTWO' , 'TEMPORARY' ); endline()
    if( line(19) and getcond( 'GCDCOMPARE' ).giveresult() and endline() ):
        line(20); getvar('LIST','GCDLIST').changeindex(getvar( 'INT' , 'ZERO' ).read()); endline()
        line(21);getvar('LIST','GCDLIST') .placevalue('INTEGERONE',"INT"); endline()
        line(22); getvar('LIST','GCDLIST').changeindex(getvar( 'INT' , 'ONE' ).read()); endline()
        line(23);getvar('LIST','GCDLIST') .placevalue('INTEGERTWO',"INT"); endline()
        updatelinewithcall( 'LIST' , 'GCDLIST' , GCD , 'LIST' , 'GCDLIST' , 24 )
    line(25); return(updatelineexitingcall( 'LIST' , 'GCDLIST' ))
updatelinewithcall( 'LIST' , 'GCDLIST' , GCD , 'LIST' , 'GCDLIST' , 26 )
printvars()