// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// EUROPEAN COMMISSION LIBRARY
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// COMPONENTS ::: POPUP
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// This plugins can be display like tooltip or modal window.
//	- language coverage popup
//	- media gallery
//  - multi-page
//  - calendar
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// pst.popup.show({
//	content		: "",		// inner content html of modal window
//	pClass		: "",		// (optional) additionnal class by user
//	pWidth		: 400,		// (optional) width of POPUP
//	pHeight		: 300,		// (optional) height of POPUP
//	overlayer	: false,	// (optional) mask background (true|false)
//	modal		: false,	// (optional) display or not like modal window other else display under cursor
//	modalTitle	: "",		// (optional) title of the modal window
// 	onClose		: function(){return false;}	// Bind event on close modal window
// })
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

pst.popup = function()
{
	var pp,pc,px,py,ps,pw,pt,pa,pe,pf,ax,ay;

	return {

		// --------------------------------------------
		// DEFAULT PARAMS
		// --------------------------------------------

		config: {

			content		: "",
			pClass		: "",
			pWidth		: 400,
			pHeight		: 300,
			overlayer	: false,
			modal		: false,
			modalTitle	: "",
			onClose		: function(){return false;}
		},

		// --------------------------------------------
		// DISPLAY "POPUP"
		// --------------------------------------------
		// i : array of config
		// --------------------------------------------

		show: function(i)
		{
			if(!document.getElementById)
			{
				return;
			}

			cfg = pst.getConfig(i,pst.popup.config);

			if( cfg["content"] == "" )
			{
				return;
			}

			pw = cfg["pWidth"];
			ph = cfg["pHeight"];

			pc = "euModalWindow"; 	// id
			px = -2; 				// adjust left position from cursor click
			py = -5; 				// adjust left position from cursor click
			ps = 10; 				// speed
			pt = 10; 				// timer
			pa = 0;					// start alpha
			pe = 100; 				// end alpha
			pf = null; 				// callback function on finish display
			ax = 0;
			ay = 0;

			document.popup 	= false;

			// --------------------------------------------
			// CREATE OR GET OVERLAYER
			// --------------------------------------------

			ov = document.getElementById("euModalWindowOverlayer");

			if( !ov && cfg["overlayer"] == true )
			{
				ov 				= document.createElement('div');
				ov.className 	= "euModalWindowOverlayer";
				ov.id 			= "euModalWindowOverlayer";

				document.body.appendChild(ov);
			}

			// --------------------------------------------
			// RESIZE OVERLAYER
			// --------------------------------------------

			if( cfg["overlayer"] == true )
			{
				docH 				= euDocType.scrollHeight;
				ov.style.display	= "";
				ov.style.width		= "100%";
				ov.style.height		= docH+"px";
			}

			// --------------------------------------------
			// GET "POPUP"
			// --------------------------------------------

			pp = document.getElementById("euModalWindow");

			// --------------------------------------------
			// KILL IF EXIST "POPUP"
			// --------------------------------------------

			if(pp)
			{
				pst.remove(pp);
			}

			// --------------------------------------------
			// CREATE "POPUP/MODAL"
			// --------------------------------------------

			pp 		= document.createElement('div');
			pp.id 	= pc;
			pp.cfg 	= cfg;

			document.body.appendChild(pp);

			// --------------------------------------------
			// CACHE ORIGINAL CSS WIDTH
			// --------------------------------------------

			if( cfg["modal"] == false )
			{
				pp = document.getElementById(pc);
			}

			// --------------------------------------------
			// ADD SPECIAL CLASS
			// --------------------------------------------

			pp.className = (cfg["pClass"]) ? ""+cfg["pClass"] : "euModalWindow";

			// --------------------------------------------
			// SET SIZE - POPUP
			// --------------------------------------------

			header = "";

			if( cfg["modal"] == false )
			{
				pp.style.width 	  	= pw+"px";
				pp.style.height		= "auto";
			}

			// --------------------------------------------
			// SET HEADER OF MODAL
			// --------------------------------------------

			else
			{
				header  = '<div class="euModalHeader">';
				header += '<div class="euModalClose euCloseModal"><span class="euCloseModal">X</span></div>';
				header += '<div id="euModalTitle" class="euModalTitle">'+cfg["modalTitle"]+'</div>';
				header += '</div>';

				pp.style.width 	  	= pw+"px";
				pp.style.height		= ph+"px";
			}

			// --------------------------------------------
			// POSITION FIRST (IE)
			// --------------------------------------------

			pp.style.display 	= "";
			pp.style.position 	= "absolute";
			pp.style.left 		= "0";
			pp.style.top 		= "0";
			pp.style.visibility	= "hidden";

			// --------------------------------------------
			// FILL "CONTENT"
			// --------------------------------------------

			c  = header
			c += "<div id='euModalContent' class='euModalContent'><div class='euIn'>";
			c += cfg["content"];
			c += "</div></div>";

			pp.innerHTML = c;

			// WAIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIT IEEEEE

setTimeout(function(){

			tz = document.getElementById("euModalContent");

			// --------------------------------------------
			// IT'S MOVIE ? -> RESIZE TO FIT MODAL ?
			// --------------------------------------------
			// 400x300

			mov = tz.getElementsByTagName("embed")[0];

			if(mov)
			{
				hd = document.getElementById("euModalTitle");
				if( hd )
				{
					hd = pst.getModelBox(hd); // modal header
				}
				mp = pst.getModelBox(pp); // modal popup
				mc = pst.getModelBox(tz); // modal content

				eh = (hd) ? hd.OH : 0;

				wm = parseInt( ((mp.PL+mp.PR)) + ((mc.BL+mc.BR+mc.PL+mc.PR+mc.ML+mc.MR)) )
				wh = parseInt( ((mp.PT+mp.PB)) + ((mc.BT+mc.BB+mc.PT+mc.PB+mc.MT+mc.MB)) + eh )

				pw = parseInt(pw)+(wm);
				ph = parseInt(ph)+(wh);

				pp.style.width 	= pw+"px";
				pp.style.height = ph+"px";

			}

			// --------------------------------------------
			// POSITION POPUP "UNDER" MOUSE
			// --------------------------------------------

			if( cfg["modal"] == false )
			{

				tz.style.height		= "auto";
				tw					= tz.offsetWidth;
				th					= tz.offsetHeight;
				pp.style.width 		= tw+"px";

				// --------------------------------------------
				// POSITION "UNDER" MOUSE
				// --------------------------------------------
				
				if( !document.keyCoord)
				{
					if((euMouse.x+tw)>euWindow.w)
					{
						ax = (euWindow.w-tw)-10;
					}
					if((euMouse.y+th)>euWindow.h)
					{
						ay = (euWindow.h-th);
					}
	
					ax = (ax) ? ax : (euMouse.x)-10;
					ay = (ay) ? ay : (euMouse.y)-10;
	
					px = (ax + px);
					py = (ay + py);
				}
				else
				{
					px = document.keyCoord[0];
					py = document.keyCoord[1];
					
					if( ( px + tw ) > euWindow.w )
					{
						px = ( euWindow.w - tw );
					}
					if( ( py + th ) > euWindow.h )
					{
						py = ( euWindow.h - th );
					}
				}

				pp.style.left	= px + "px";
				pp.style.top	= py + "px";
				
				// --------------------------------------------
				// REPLACE IF SO TOP
				// --------------------------------------------

				if( th > euDocType.clientHeight )
				{
					pp.style.height = (euDocType.clientHeight)-35+"px";
					tz.style.height = (euDocType.clientHeight)-35+"px";
					tz.style.width 	= (tw+16)+"px";

					px = parseInt(pp.style.left)-10;
					py = (euScroll.y)+10;

					pp.style.top 	= py+"px";
					pp.style.left 	= px+"px";
				}

				// --------------------------------------------
				// REPLACE IF SO LEFT
				// --------------------------------------------

				if( px < 10 )
				{
					px = 15;
					pp.style.left 	= px + "px";
				}

				document.popup = true;

			}

			// --------------------------------------------
			// CENTER MODAL
			// --------------------------------------------

			else
			{
				pp.style.top  = ( ( (euDocType.clientHeight/2)-(ph/2) ) + euScroll.y )+"px";
				pp.style.left = ( ( (euDocType.clientWidth/2)-(pw/2) ) + euScroll.x )+"px";
			}

			// --------------------------------------------
			// DISPLAY IT -> FADE IN OR OUT
			// --------------------------------------------

			pp.style.visibility	= "visible";

			// --------------------------------------------
			// TRY TO FOCUS ON FIRST LINK ?
			// --------------------------------------------
			
			setTimeout(function(){
				
				firstLinks = pp.getElementsByTagName("A")[0];
	
				if( firstLinks && document.keyCoord )
				{
					pst.keyboard.setNav(pp);
					document.keyCoord = null;
					
				}
				else
				{
					pp.focus();
				}
				
			},100); // IE7

			// --------------------------------------------
			// HIDE ON MOUSE OUT POPUP
			// --------------------------------------------

			if( cfg["modal"] == false )
			{
				minLeft = px;
				maxLeft = Math.round( px + pp.offsetWidth );
				minTop  = py;
				maxTop  = Math.round( py + pp.offsetHeight );

				pp.position = {minLeft:minLeft,maxLeft:maxLeft,minTop:minTop,maxTop:maxTop};

				pst.addEvent(pp,"mouseover",pst.popup.bindOver);
			}

			pst.addEvent(pp,"keydown",pst.popup.key);


},100); // IE

		},

		bindOver: function()
		{
			pst.addEvent(document,"mousemove",pst.popup.checkOut);
		},

		// --------------------------------------------
		// PRESS ESC -> HIDE POPUP AND FOCUS ON LAST ENTRY
		// --------------------------------------------

		key: function(event)
		{
			pst.captureEvent(event);

			if( keyCode == 27 )
			{
				if( pst.lastFocus )
				{
					pst.lastFocus.focus();
				}

				pst.popup.hide();
			}

		},

		// --------------------------------------------
		// CHECK IF MOUSE IS OUT OR NOT
		// --------------------------------------------

		checkOut: function(event)
		{
			pst.captureEvent(event);

			if(!pp.position){return;}

			minLeft = pp.position.minLeft;
			maxLeft = pp.position.maxLeft;
			minTop  = pp.position.minTop;
			maxTop  = pp.position.maxTop;

			if( euMouse.x < minLeft || euMouse.x > maxLeft || euMouse.y < minTop || euMouse.y > maxTop )
			{
				pst.popup.hide();
				pst.removeEvent(document,"mousemove",pst.popup.checkOut);
				pst.removeEvent(this,"mouseover",pst.popup.bindOver);

				if( pst.lastFocus )
				{
					pst.lastFocus.focus();
				}

			}
		},

		pos:function(x,y)
		{
			pp = document.getElementById("euPopup");
			if(pp)
			{
				pp.style.left	= x + "px";
				pp.style.top	= y + "px";
			}
		},

		hide:function(evt)
		{
			if(pp)
			{
				cfg = pp.cfg;
				cfg["onClose"]();
				pp.style.display = "none";
				pp.innerHTML = " ";

				pst.keyboard.unSetNav(pp);

				pst.removeEvent(pp,"keydown",pst.popup.key);

				if( pst.lastFocus )
				{
					pst.lastFocus.focus();
				}
			}
			if(ov) // overlayer
			{
				ov.style.display="none";
			}

			document.popup = false;

		},

		isShow:function()
		{
			pp = document.getElementById(pc);
			if( pp && pp.style.display == "block"  )
			{
				pp.style.display = "none";
			}
		}

	};

}();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ADD EVENT CLICK
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pst.addEvent(document,"click",function(evt)
{
	pst.captureEvent(evt);

	// ---------------------------------------------
	// HIDE CURRENT MODAL
	// ---------------------------------------------

	if( ( euElement.tagName == 'DIV' || euElement.tagName == "SPAN" ) && euElement.className.indexOf('euCloseModal') != -1  )
	{
		pst.popup.hide();
	}

	// ---------------------------------------------
	// HIDE CURRENT POPUP
	// ---------------------------------------------

	else if( document.popup == true  )
	{
		setTimeout(function(){ // IE
			pst.popup.hide();
		},100);
	}

});