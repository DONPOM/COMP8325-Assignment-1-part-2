// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// EUROPEAN COMMISSION "New Toggle Box" 0.0.1
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Author Braije Christophe 2008
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

var cmScroll	= {x:0,y:0};
var cmMouse		= {x:0,y:0};
var cmWindow	= {w:0,h:0};
var cmDocType	= (document.compatMode && document.compatMode != "BackCompat")? document.documentElement : document.body;

function cmCaptureEvent(evt)
{	
	if(!evt){evt=window.event;} // IE
	var cmDocType	= (document.compatMode && document.compatMode != "BackCompat")? document.documentElement : document.body; //NICO: copy of line #10
	
	cmScroll.x	= document.all ? cmDocType.scrollLeft : pageXOffset;
	cmScroll.y	= document.all ? cmDocType.scrollTop : pageYOffset;
	cmMouse.x 	= evt.clientX+cmScroll.x;
	cmMouse.y 	= evt.clientY+cmScroll.y;
	cmWindow.w 	= cmDocType.clientWidth+cmScroll.x;
	cmWindow.h 	= cmDocType.clientHeight+cmScroll.y;
}

function cmAddEvent(o,e,f)
{
	if ( o.addEventListener ) // W3C
	{
		o.addEventListener(e,f,false);
	}
	else if ( o.attachEvent ) // IE
	{
		o.attachEvent('on'+e,f);
	}
}

function toggleBox( obj, lyr, iState, iPosition) 
{
	sr = document.getElementById(lyr);
	
	if(!sr){ return; }
	
	tb = document.getElementById("cmPopup");
	
	if(!tb)
	{
		tb					= document.createElement('div');
		tb.id				= "cmPopup";
		tb.style.position 	= "absolute";
		document.body.appendChild(tb);
	}
	
	tb.innerHTML 		= sr.innerHTML;
	tb.style.display  	= iState ? "block" : "none";
	
	setTimeout(function(){
	
		tx	= 0;
		ty	= 0;
		tw 	= 160;
		th 	= 100;
	
		if((cmMouse.x+tw)>cmWindow.w)
		{
			tx = (cmWindow.w-tw)-10;
		}
		if((cmMouse.y+th)>cmWindow.h)
		{
			ty = (cmWindow.h-th);
		}
		
		tx = (tx) ? tx : cmMouse.x;
		ty = (ty) ? ty : cmMouse.y;
		
		tb.style.left	= (tx-5) + "px";
		tb.style.top	= (ty-5) + "px";
	
		
	},10);
	
	tb.onmouseout 	= function(){ tb.timer = setTimeout(function(){tb.style.display="none";},150) };
	tb.onmouseover 	= function(){ clearTimeout(tb.timer); };
	
}

cmAddEvent(document,"click",cmCaptureEvent);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// OLD FUNCTION ( Keep for backward compatibility? )
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function ResetShortcuts()
{
   if(document.LangForm != null) {
    for (var i = 0; i < document.LangForm.Language.length; i++) {
        if (document.LangForm.Language.options[i].defaultSelected == true){
            document.LangForm.Language.options[i].selected=true
        }
    }
   }
}
function JumpMenu(item)
{
    document.location= item;
}
function findPosX(obj)
{
	var curleft = 0;
	if (obj.offsetParent)
	{
		while (obj.offsetParent)
		{
			if(obj.style.position != 'relative'){
			curleft += obj.offsetLeft;
			}
			obj = obj.offsetParent;
		}
	}
	else if (obj.x)
		curleft += obj.x;
	return curleft;
}
function findPosY(obj)
{
	var curtop = 0;
	if (obj.offsetParent)
	{
		while (obj.offsetParent)
		{
			if(obj.style.position != 'relative'){
				curtop += obj.offsetTop;
			}
			obj = obj.offsetParent;
		}
	}
	else if (obj.y)
		curtop += obj.y;
	return curtop;
}
function getObj(name)
{
 if (document.getElementById)
 {
	   this.obj = document.getElementById(name);
	   this.style = document.getElementById(name).style;
 }
 else if (document.all)
 {
	   this.obj = document.all[name];
	   this.style = document.all[name].style;
 }
 else if (document.layers)
 {
	   if (document.layers[name])
	   {
	   	this.obj = document.layers[name];
	   	this.style = document.layers[name];
	   }
	   else
	   {
	    this.obj = document.layers.testP.layers[name];
	    this.style = document.layers.testP.layers[name];
	   }
 }
}