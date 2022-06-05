



<!-- Ticker Tape in Java Script .. Heavily Modified by Bill Welliver



//-- Original concept by Cameron Gregory cameron@corona.att.com



var tickertapeform



tt_speed=150



tt_len=35



tt_space="                                                                                                    "



tt_tid = 0;



tt_message="."



tt_c= -tt_len;







function move() {



  cend=Math.min(tt_c+tt_len,tt_message.length)



  if (tt_c <0)



    cstart=0;



  else



    cstart=tt_c;



  if (tt_c < 0)



    tt_f.scroll.value=tt_space.substring(0,-tt_c) + tt_message.substring(cstart,cend)



  else



    tt_f.scroll.value=tt_message.substring(cstart,cend)



  tt_c = tt_c +1;



  if (tt_c == tt_message.length ) tt_c = -tt_len;



  tt_tid=window.setTimeout("move()",tt_speed);



}



 



function tt_start(x) {



  tt_f=x



  tt_tid=window.setTimeout("move()",tt_speed);



}



 



function tt_cleartid() {



  window.clearTimeout(tt_tid);



}



 



// for some reason on some pages this crashes netscape



function ticker(m,l,s)



{



tt_message=m



tt_len=l



tt_speed=s



document.write('<FORM name=tickertapeform><input name=scroll size=')



document.write(tt_len)



document.write(' value=""></FORM>')



tt_start(document.tickertapeform);



}







// for some reason on some pages this crashes netscape



function ticker(m)



{



tt_message=m



tt_len=35



tt_speed=150



document.write('<FORM name=tickertapeform><input name=scroll size=35></FORM>');



tt_start(document.tickertapeform);



}



// end-->



