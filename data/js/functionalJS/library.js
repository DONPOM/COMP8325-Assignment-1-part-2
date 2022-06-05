// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// EUROPEAN COMMISSION LIBRARY
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

var pst = {

	// ---------------------------------------------
	// CONFIG
	// ---------------------------------------------

	path: "/wel/template-2009/scripts/",

	// ---------------------------------------------
	// BROWSER DETECT
	// ---------------------------------------------

	browser: {

		isIE 	: /*@cc_on!@*/false,
		isIE6 	: false /*@cc_on || @_jscript_version < 5.7 @*/

	},

	// ---------------------------------------------
	// PROCESS ORDER OF PST
	// ---------------------------------------------

	run: function()
	{
		// ---------------------------------------------
		// AVOID VULNERABILITY IE
		// ---------------------------------------------
		// http://labs.idefense.com/intelligence/vulnerabilities/display.php?id=77
		// ---------------------------------------------

		if( top != self )
    	{
        	top.location = self.location;
    	}

    	// ---------------------------------------------
    	// DOM ACCESS FROM DOCTYPE TYPE
    	// ---------------------------------------------

		euDocType 	= (document.compatMode && document.compatMode != "BackCompat")? document.documentElement : document.body;

		// ---------------------------------------------
		// hack - css overflow-y is not W3C
		// force to display always the scrollbar Y
		// ---------------------------------------------

		euDocType.style.overflowY = "scroll";
		
		// ---------------------------------------------
		// PROCESS ORDER
		// ---------------------------------------------

		pst.minMaxCSS();			// IE6< hack css
		pst.accessibility.init();	// print, font increase ...
		
		setTimeout(function(){
			pst.translate.init();
			pst.menu.init();
			pst.webservice.init();
			pst.components.init();
		},10); // IE7

	},

	// ---------------------------------------------
	// AS SOON DOM IS READY
	// ---------------------------------------------
	// pst.ready(functionName);
	// pst.ready(function(){});
	// ---------------------------------------------

	ready: function(func)
	{
	    if( pst.domIsReady )
	    {
	    	return func();
	    }

		if(!pst.loadEvents)
		{
			pst.loadEvents = [];
		}

	    if( !pst.loadEvents[0] )
	    {
	        if( document.addEventListener ) // for Mozilla/Opera9
	        {
	            document.addEventListener("DOMContentLoaded", pst.allReady, false);
	        }
	        else if( pst.browser.isIE ) // for Internet Explorer
	        {
	            document.write("<script id=__ie_onload defer src=//0><\/scr"+"ipt>");
	            pst.ieReady = document.getElementById("__ie_onload");
	           	pst.ieReady.onreadystatechange = function()
	            {
	                if( this.readyState == "complete" )
	                {
	                    pst.allReady();
					}
	            };
	        }
	        else if (/WebKit/i.test(navigator.userAgent)) // for Safari
	        {
	            pst.loadTimer = setInterval(function()
	            {
	                if (/loaded|complete/.test(document.readyState))
	                {
	                    pst.allReady();
	                    clearInterval(pst.loadTimer);
	                }
	            }, 10);
	        }

			// old browser
	        pst.oldOnload = window.onload;

	        window.onload = function()
	        {
	            pst.allReady();
	            if( pst.oldOnload )
	            {
	            	pst.oldOnload();
	            }
	        };
	    }

	    pst.loadEvents.push(func);
	},

	allReady: function()
	{
		pst.domIsReady  = true;

		clearInterval(pst.loadTimer);

		while( pst.exec = pst.loadEvents.shift() )
		{
			pst.exec();
		}
		if( pst.ieReady )
		{
			pst.ieReady.onreadystatechange = '';
		}
	},

	// ---------------------------------------------
	// CAPTURE EVENT
	// ---------------------------------------------

	captureEvent: function(event)
	{
		event			= event || window.event;
		euElement 		= event.target || event.srcElement;
		euElement.event = event;

		keyCode			= null;

		if( /KEYPRESS|KEYDOWN|KEYUP/i.test(event.type) )
		{
			keyCode = (event.keyCode ? event.keyCode: (event.charCode ? event.charCode: event.which));
		}

		euDocType 	= (document.compatMode && document.compatMode != "BackCompat")? document.documentElement : document.body;
		euScroll 	= {x: pst.browser.isIE ? euDocType.scrollLeft : pageXOffset, y: pst.browser.isIE ? euDocType.scrollTop : pageYOffset};
		euMouse 	= {x: event.clientX + euScroll.x, y: event.clientY + euScroll.y};
		euWindow 	= {w: euDocType.clientWidth + euScroll.x, h: euDocType.clientHeight + euScroll.y};

	},

	// ---------------------------------------------
	// ON RESIZE (HACK IE)
	// ---------------------------------------------
	// pst.resize(functionName);
	// pst.resize(function(){});
	// ---------------------------------------------
	// CALL THIS FUNCTION AFTER LOAD EVENT ! IE5.5
	// ---------------------------------------------

	resize: function( func )
	{
		if(!pst.isResize)
		{
			pst.isResize = [];
		}
		if( pst.browser.isIE )
		{
			pst.isResize[func] = pst.getViewport();
		}

		var v,h,w,o,ow,oh;

		return pst.addEvent(window,"resize",function(){

			if( pst.browser.isIE )
			{
				v 	= pst.getViewport();
				h 	= v.h;
				w 	= v.w;
				o 	= pst.isResize[func];
				ow	= o.w;
				oh	= o.h;

				pst.isResize[func] = {w:w,h:h};

				if( oh != h || ow != w )
				{
					if( typeof func === "function" )
					{
						func();
					}
				}
			}
			else if( typeof func === "function" )
			{
				func();
			}

		},false);

	},

	// ---------------------------------------------
	// Bind event on object and avoid overide
	// on existe binding function
	// ---------------------------------------------
	// pst.addEvent(window,"load",functionName);
	// pst.addEvent(document,"click",functionName);
	// ---------------------------------------------

	addEvent: function(elem, eventType, handler)
	{
		if( !elem )
		{
			return;
		}

		if( !elem.eventHandlers )
		{
			elem.eventHandlers = [];
		}

		if( !elem.eventHandlers[eventType] )
		{
			elem.eventHandlers[eventType] = [];

			if( elem['on' + eventType])
			{
				elem.eventHandlers[eventType].push(elem['on' + eventType]);
			}
			elem['on' + eventType] = pst.handleEvent;
		}
		elem.eventHandlers[eventType].push(handler);

	},

	// ---------------------------------------------
	// Remove event on object
	// ---------------------------------------------
	// pst.removeEvent(window,"load",functionName);
	// pst.removeEvent(document,"click",functionName);
	// ---------------------------------------------

	removeEvent: function( elem, eventType, handler)
	{
		if( elem.eventHandlers )
		{
			var handlers = elem.eventHandlers[eventType];
			
			for(var i in handlers)
			{
				if (handlers[i] == handler)
				{
					delete handlers[i];
				}
			}
		}
	},

	// ---------------------------------------------
	// Internal event function
	// ---------------------------------------------

	handleEvent: function(e)
	{
		var returnValue = true,i,h;
		if( !e )
		{
			e = e || window.event;
			e = pst.fixEvent(e);
		}
		h = this.eventHandlers[e.type];

		for (var i in h)
		{
			if (isNaN (i))
			{
				continue;
			}

			this.$$handleEvent = h[i];

			if( typeof this.$$handleEvent != "function")
			{
				continue;
			}
			if( this.$$handleEvent(e) === false )
			{
				returnValue = false;
			}
		}

		return returnValue;
	},

	fixEvent: function(e)
	{
		if(!e){return false;}
		// add W3C standard event methods
		e.preventDefault = pst.stopDefault;
		e.stopPropagation = pst.cancelBubbling;
		return e;
	},

	stopDefault: function(e)
	{
		e = e || window.event;
		e.returnValue = false;
		if(e.preventDefault)
		{
			e.preventDefault();
		}
	},

	cancelBubbling: function(e)
	{
		e = e || window.event;
		e.cancelBubble = true;
		if(e.stopPropagation)
		{
			e.stopPropagation();
		}
	},

	// ---------------------------------------------
	// ADD NEW COLLECTION OF FUNCTIONS IN PST
	// ---------------------------------------------
	// pst.extend({
	//		functionName1 : function(){ .... },
	//		functionName2 : function(){ .... }
	// })
	// ---------------------------------------------

	extend: function()
	{
		var t = arguments[0] || {}, i = 1, length = arguments.length, deep = false, options;

		if ( typeof t === "boolean" )
		{
			deep = t;
			t = arguments[1] || {};
			i = 2;
		}

		if ( typeof t !== "object" )
		{
			t = {};
		}

		if ( length == i )
		{
			t = this;
			--i;
		}

		for ( ; i < length; i++ )
		{
			if ( (options = arguments[ i ]) !== null )
			{
				for ( var name in options )
				{
					var src = t[ name ], copy = options[ name ];
					if ( t === copy )
					{
						continue;
					}

					if ( deep && copy && typeof copy === "object" && !copy.nodeType )
					{
						t[ name ] = pst.extend( deep, src || ( copy.length !== null ? [ ] : { } ), copy );
					}
					else if ( copy !== undefined )
					{
						t[ name ] = copy;
					}
				}
			}
		}
		return t;
	},

	// ---------------------------------------------
	// INSERT ELEMENT JUST AFTER ANOTHER
	// ---------------------------------------------
	// n : newElement
	// t : targetElement
	// ---------------------------------------------

	after: function(n,t)
	{
		if(t)
		{
			var p = t.parentNode;

			if( p.lastchild == t )
			{
				p.appendChild(n);
			}
			else
			{
				p.insertBefore(n, t.nextSibling);
			}
		}
	},

	// ---------------------------------------------
	// INSERT ELEMENT JUST BEFORE ANOTHER
	// ---------------------------------------------
	// n : newElement
	// t : targetElement
	// ---------------------------------------------

	before: function( n, t )
	{
		if( t )
		{
			var p = t.parentNode;

			if( p )
			{
				p.insertBefore( n, t );
			}
		}
	},

	// ---------------------------------------------
	// Removes an element from the DOM.
	// ---------------------------------------------
	// e : element reference object
	// ---------------------------------------------

	remove: function( e )
	{
        if( e )
        {
        	if( e.parentNode )
        	{
        		e.parentNode.removeChild( e );
        	}
        }
    },

	// ---------------------------------------------
	// CREATE XMLHttpRequest
	// ---------------------------------------------

	xhr: function()
	{
		var x = false;
		if( window.XMLHttpRequest ) /* Mozilla/Safari */
		{
			x = new XMLHttpRequest();
		}
		else if( window.ActiveXObject ) /* Internet Explorer */
		{
			x = new ActiveXObject("Microsoft.XMLHTTP");
   		}
   		else
   		{
   			// futur iframe version ?
   		}
		return x;
	},

	// ---------------------------------------------
	// RETURN TEXT AND XML DATA FROM FILE LOADING
	// ---------------------------------------------
	// PARAMS
	// ---------------------------------------------
	// 	url		: (mandatory) url of file to load
	// 	success	: (optional) callback function on success load.
	// 	error	: (optional) callback function on error
	// 	other	: (optional) All others data are transfer to succes and error function callback.
	// ---------------------------------------------
	// EXAMPLE
	// ---------------------------------------------
	// 	pst.load({
	//		url 	: "my_url",
	//		success : function(responseText,responseXML,myData){ alert("My content is load"); },
	//		error	: function(myData){ alert("Content not found !"); },
	// 		myData	: "any information",
	//  	other	: function(){ alert("This my personnal function"); }
	// 	})
	// ---------------------------------------------

	load: function(c)
	{
		var u = c.url;

		if( u !== "" && u !== undefined && u !== null )
		{
			var r = pst.xhr();

			if(!r){return;}

			if( typeof c.start == "function" )
    		{
    			c.start(c);
    		}

    		u = u.replace(/&amp;/ig,"&");

			r.onreadystatechange = function()
			{
			    if( r.readyState == 4 )
			    {
			    	if(r.status != 200 && r.status != 304)
			    	{
			    		if( typeof c.error == "function" )
			    		{
			    			c.error(c);
			    		}
			    	}
			    	else
			    	{
			    		if( typeof c.success == "function" )
				    	{
				        	 c.success(r.responseText,r.responseXML,c);
				        }
				        else
				        {
				        	return {txt:r.responseText,xml:r.responseXML};
				        }
				    }
			    }
			};

			r.open("GET",u, true);
			r.send(null);
		}
	},

	// ---------------------------------------------
	// TOOLS ::: LOAD JAVASCRIPT OR CSS AND EXECUTE CALlBACK
	// ---------------------------------------------
	// u : url of file
	// c : callback onload function
	// ----------------------------------------------
	// pst.include("http://www.domaine.com/myScript.js");
	// pst.include("http://www.domaine.com/myCss.css");
	// pst.include("/mypath/myScript.js");
	// pst.include("/mypath/myScript.js",function(){});
	// ----------------------------------------------

	include: function(u,c)
	{
		if(!pst.isLoad)
		{
			pst.isLoad = {};
		}

		var i,t,e;

		t = ( pst.isLoad[u] ) ? true : false;

		if( t === false )
		{
			pst.isLoad[u] = u;

			e = pst.getExtention(u);

			if( e == "js" )
			{
				i=document.createElement('script');
				i.setAttribute('type','text/javascript');
				i.setAttribute('src',u);
			}
			else if( e == "css" )
			{
				i=document.createElement('link');
				i.setAttribute('type','text/css');
				i.setAttribute('rel','stylesheet');
				i.setAttribute('media','all');
				i.setAttribute('href',u);
			}

			if( typeof c == "function" )
			{
				if( pst.browser.isIE )  // IE
				{
					i.onreadystatechange = function()
	                {
	                    if( /loaded|complete/ig.test(this.readyState) )
	                    {
                        	c();
						}
	                };
				}
				else // W3C
				{
					i.onload = c;
				}
			}

			if( e == "css" )
			{
				h = document.getElementsByTagName('head')[0];
				if(h){h.appendChild(i);}
			}
			else if( e == "js" )
			{
				b = document.getElementsByTagName('body')[0];
				if(b){b.appendChild(i);}
			}
		}

		else if( typeof c == "function" )
		{
			c();
		}
		return;

	},

	// ---------------------------------------------
	// RETURN ARRAY OF w(idth) AND h(eight) OF WINDOW VIEWPORT
	// ---------------------------------------------

	getViewport: function()
	{
		return {
			w : (!window.innerWidth) ? euDocType.clientWidth : window.innerWidth,
			h : (!window.innerHeight) ? euDocType.clientHeight : window.innerHeight
		};
	},

	// ---------------------------------------------
	// RETURN THE ABSOlUTE POSITION OF ELEMENT
	// ---------------------------------------------

	getPosition: function( domElm )
	{
		var x,y;
		
		x = 0;
		y = 0;

		if( domElm.offsetParent )
		{
			do {
				x += domElm.offsetLeft;
				y += domElm.offsetTop;  
			} while (domElm = domElm.offsetParent);
		}
		return [x,y];
	},

	// ---------------------------------------------
	// RETURN EXTENTION FROM "STRING" (URL)
	// ---------------------------------------------

	getExtention: function( myURL )
	{
		if(!myURL){return;}
		var myURL = myURL.toLowerCase();
		return (/[.]/.exec(myURL)) ? /[^.]+$/.exec(myURL) : undefined;
	},

	// ---------------------------------------------
	// SIMPLE STORE COOKIE
	// ---------------------------------------------
	// n : name of cookie to store
	// v : value of cookie
	// d : number of days to store cookie (optional)
	// ---------------------------------------------

	setCook: function( n, v, d)
	{
		var s,e="";

		if( d )
		{
			s = new Date();
			s.setTime(s.getTime()+(d*24*60*60*1000));
			e = ";expires="+s.toGMTString();
		}
		document.cookie = n+"="+v+e+";path=/";
	},

	// ---------------------------------------------
	// SIMPLE GET COOKIE
	// ---------------------------------------------
	// n : name of cookie to retrieve
	// d : default value if cookie not found (optional)
	// ---------------------------------------------

	getCook: function( n, d )
	{
		var c,o,n,i,t;

		o = document.cookie.split(';');
		n = n + "=";

		for(i = 0, t = o.length; i < t; i++)
		{
			c = o[i];

			while ( c.charAt(0) == ' ' )
			{
				c = c.substring(1,c.length);
			}

			if( c.indexOf(n) === 0 )
			{
				return c.substring(n.length,c.length);
			}
		}
		return d||null;
	},

	// ---------------------------------------------
	// SIMULATE THE MAX-WIDTH AND MIN-WIDTH FOR IE6<
	// ---------------------------------------------

	minMaxCSS: function()
	{
		var layout = document.getElementById("euLayout");

		if( pst.browser.isIE6 && layout )
		{
			var minWidth = pst.minWidth||null;
			var maxWidth = pst.maxWidth||null;

			if( minWidth === null || maxWidth === null )
			{
				pst.minWidth = parseInt(layout.currentStyle.minWidth || layout.currentStyle["min-width"],10)||0;
				pst.maxWidth = parseInt(layout.currentStyle.maxWidth || layout.currentStyle["max-width"],10)||"auto";

				pst.resize(pst.minMaxCSS);
				
				setTimeout(function(){
					pst.minMaxCSS();
				},250);
					
			}
			else
			{
				layout.style.width  = (euDocType.clientWidth < (minWidth+2) ? minWidth+"px" : (euDocType.clientWidth > (maxWidth+2) && maxWidth != "auto" ) ? maxWidth+"px" : "auto");
			}
		}
	},

	// ---------------------------------------------
	// WORKAROUND TO HIDE ALL EMBED
	// ---------------------------------------------
	// v : hidden|visible
	// ---------------------------------------------

	hideEmbed: function( v )
	{
		var t,a,i;

		a = document.body.getElementsByTagName("EMBED");

		for(i = 0, t = a.length; i < t; i++)
		{
			a[i].style.visibility = v;
		}
	}

};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// EXTEND FUNCTION - COMPONENTS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pst.extend({
	
	// ---------------------------------------------
	// RETURN CLEAN STRING WITHOUT HTML CODE
	// ---------------------------------------------
	// s : string to cleaning
	// ---------------------------------------------
	
	cleanHTML: function(s)
	{
		if(!s){return "";}
		s = s.replace(/\n|\t|\r/ig,"");
		s = s.replace(/\s\s/ig," ");
		s = s.replace(/&(lt|gt);/ig, function (m, p1){return (p1 == "lt")? "<" : ">";}); // IE !
		s = s.replace(/<\/?[^>]+(>|$)/ig, "");
	 	return s
	},
	
	// ---------------------------------------------
	// RETURN ARRAY OF "CLASS CONFIG" FROM DOM OBJECT
	// ---------------------------------------------
	// o : object dom where class is define
	// n : name of class config (string)
	// d : default config (array)
	// ---------------------------------------------

	getClassConfig: function (o,n,d)
	{
		var c,i,e,l,m,z,t,a,b,j,v;

		c = o.className;
		e = c.split(" ");
		l = e.length;
		m = [];

		for(i = 0; i < l; i++)
		{
			c = e[i];

			// ---------------------------------------------
			// string is like -> class[name:value,name:value]
			// ---------------------------------------------

			if( c.indexOf(n) != -1 && c.indexOf(":") != -1 )
			{
				c = c.replace(n+"[","").replace("]","");
				c = c.split(",");
				n = [];

				for(z = 0, t = c.length; z < t; z++)
				{
					a = c[z].split(":");
					b = a[0];
					j = a[1];
					n[b] = j;
				}

				m = n;
				break;
			}

			// ---------------------------------------------
			// string is like -> class[value1,value2]
			// ---------------------------------------------

			else if( c.indexOf(n+"[") != -1 )
			{
				c = c.replace(n+"[","").replace("]","");
				c = c.split(",");
				m = c;
				break;
			}
		}

		if( !m )
		{
			m = d;
		}
		else
		{
			for(v in d)
			{
				m[v] = (m[v]) ? m[v] : d[v];
			}
		}

		return m;

	},

	// ---------------------------------------------
	// OVERWRITE ARRAY[d] BY ARRAY[u]
	// ---------------------------------------------
	// u : user config (array)
	// d : default config (array)
	// ---------------------------------------------

	getConfig: function(u,d)
	{
		var z,i;

		z=[];

		for(i in d)
		{
			z[i] = (u[i]) ? u[i] : d[i];
		}
		return z;
	},

	// ---------------------------------------------
	// TOOLS ::: RETURN AN ARRAY OF DOM WITH CLASS
	// ---------------------------------------------
	// n : node
	// t : tag
	// c : search class
	// ---------------------------------------------

	getByClass: function(n,t,c)
	{
		var b,r,a,i,j,e; // ;-)

		if( n === null )
		{
			n = document;
		}
		if( t === null )
		{
			t = '*';
		}

		a = [];
		e = n.getElementsByTagName(t);
		b = e.length;
		r = new RegExp("(^|\\s)"+c+"(\\s|$)");

		for(i = 0, j = 0; i < b; i++)
		{
			if( r.test(e[i].className) )
			{
				a[j] = e[i];
				j++;
			}
		}
		return a;
	},

	// ---------------------------------------------
	// GET SPECIFIC CSS RULES FROM ELEMENT
	// ---------------------------------------------
	// e : element
	// a : css attribute
	// d : default value
	// ---------------------------------------------

	getCSS: function(e,a,d)
	{
		if( typeof e != "object")
		{
			return d;
		}

		var v,n;

		if( window.getComputedStyle )
		{
			v = window.getComputedStyle(e,null).getPropertyValue(a);
		}
		else if( e.currentStyle )
		{
			while( a.indexOf('-') != -1 )
			{
				n = a.charAt(a.indexOf('-')+1);
				a = a.replace(/-\S{1}/,n.toUpperCase());
			}
			v = e.currentStyle[a];
		}
		if( v )
		{
			v = v.replace(/^\s+/,'').replace(/\s+$/,'').replace(/(px)?(#)?/ig,'');
			if( v.indexOf('rgb(') != -1 )
			{
				v = pst.rgb2Hex(v);
			}
		}
		if(isNaN(v)){v=d;}
		return (v!="auto")?v:d;
	},

	// ---------------------------------------------
	// GET GLOBAL CSS RULES FROM ELEMENT
	// ---------------------------------------------
	// domElm : element object
	// ---------------------------------------------

	getModelBox: function( domElm )
	{
		if( typeof domElm !== "object" )
		{
			return;
		}

		var BT,BR,BB,BL,PT,PR,PB,PL,MT,MR,MB,ML,OW,OH,IW,IH;

		BT = parseInt(pst.getCSS(domElm,"border-Top-Width","0"),10);
		BR = parseInt(pst.getCSS(domElm,"border-Right-Width","0"),10);
		BB = parseInt(pst.getCSS(domElm,"border-Bottom-Width","0"),10);
		BL = parseInt(pst.getCSS(domElm,"border-Left-Width","0"),10);

		PT = parseInt(pst.getCSS(domElm,"padding-Top","0"),10);
		PR = parseInt(pst.getCSS(domElm,"padding-Right","0"),10);
		PB = parseInt(pst.getCSS(domElm,"padding-Bottom","0"),10);
		PL = parseInt(pst.getCSS(domElm,"padding-Left","0"),10);

		MT = parseInt(pst.getCSS(domElm,"margin-Top","0"),10);
		MR = parseInt(pst.getCSS(domElm,"margin-Right","0"),10);
		MB = parseInt(pst.getCSS(domElm,"margin-Bottom","0"),10);
		ML = parseInt(pst.getCSS(domElm,"margin-Left","0"),10);

		OW = domElm.offsetWidth;
		OH = domElm.offsetHeight;

		IW = (OW-PL-PR-BL-BR);
		IH = (OH-PT-PB-BT-BB);

		return {
			OW:OW,OH:OH,IW:IW,IH:IH,
			BT:BT,BR:BR,BB:BB,BL:BL,
			PT:PT,PR:PR,PB:PB,PL:PL,
			MT:MT,MR:MR,MB:MB,ML:ML
		};
	},

	// ---------------------------------------------
	// EXTRACT URL INFO
	// ---------------------------------------------

	getUrlParam: function()
	{
		var u,x,y,n,p,t,i,l;

		u = window.location.href;
		x = new RegExp("[&?]+","g");
		y = new RegExp("[=#]+","g");
		n = u.split(x);
		p = [];

		if( n !== null )
		{
			for(i = 1, l = n.length; i < l; i++)
			{
				t = n[i].split(y);
				p[t[0]] = t[1];
			}
		}
		return p;
	},

	// ---------------------------------------------
	// RETURN RANDOM NUMBER BETWEEN min AND max
	// ---------------------------------------------
	// min : minimum number ( default = 0 )
	// max : maximum number
	// ---------------------------------------------

	rand: function( min, max )
	{
	    var a = arguments.length;

	    if( a === 0 )
	    {
	    	min = 0;
	    	max = 2147483647;
	    }
	    else if( a == 1 )
	    {
	    	max = min;
	    	min = 0;
	    }
	    return Math.floor(Math.random() * (max - min + 1)) + min;
	},

	// ---------------------------------------------
	// ANIMATE FADE/IN/OUT ELEMENT
	// ---------------------------------------------

	fade: function( id, opacStart, opacEnd, millisec, callBack)
	{
	    var i,s,t;

	    s = Math.round(millisec / 100);
	    t = 0;

	    if( opacStart > opacEnd )
	    {
	        for(i = opacStart; i >= opacEnd; i=i-2)
	        {
	            setTimeout("pst.opacity(" + i + ",'" + id + "')",(t * s));
	            t++;
	        }
	    }
	    else if( opacStart < opacEnd )
	    {
	        for(i = opacStart; i <= opacEnd; i=i+2)
	        {
	            setTimeout("pst.opacity(" + i + ",'" + id + "')",(t * s));
	            t++;
	        }
	    }

	    if( typeof callBack == "function" )
	    {
	    	setTimeout(function(){

	    		callBack();

	    	},(millisec+25));
	    }

	},

	// ---------------------------------------------
	// SET OPACITY ELEMENT
	// ---------------------------------------------

	opacity: function( opacity, elmOrID )
	{
		var e;

		e = elmOrID.style;

		if(!e)
		{
			e = document.getElementById(elmOrID);

			if(e)
			{
				e = e.style;
			}
			else
			{
				return;
			}
		}

		if( pst.browser.isIE ) // IE memory leak
		{
			e.removeAttribute('filter');
			e.filter="progid:DXImageTransform.Microsoft.Alpha(opacity="+opacity+")";
		}
		else
		{
			e.opacity 		= (opacity / 100);
    		e.MozOpacity 	= (opacity / 100);
    		e.KhtmlOpacity  = (opacity / 100);
		}
	},

	// ---------------------------------------------
	// DISPLAY OR HIDE AN ELEMENT WITH CLASS
	// ---------------------------------------------
	// USE FOR PRINT VERSION COMPONENTS
	// ---------------------------------------------

	toggleDisplay: function( domElm , showOrHide )
	{
		if( !domElm )
		{
			return;
		}

		if( pst.browser.isIE6 )
		{
			domElm.style.display = (showOrHide == "show") ? "block" : "none";
			return;
		}

		var c = domElm.className;

		if( domElm.style )
		{
			domElm.style.display = "";
		}

		if( c )
		{
			c = c.replace(/( )?(_hide_|_show_)/,"");
		}

		domElm.className = ( showOrHide == "show" ) ? c + " _show_" : c + " _hide_" ;

	},

	show: function( domElm )
	{
		pst.toggleDisplay( domElm, "show");
		return;
	},

	hide: function( domElm )
	{
		pst.toggleDisplay( domElm, "hide");
		return;
	}

});

// +++++++++++++++++++++++++++++++++++++++++++++
// PLUGINS - KEYBOARD NAVIGATION
// +++++++++++++++++++++++++++++++++++++++++++++

pst.keyboard = {

	unSetNav: function( domElm )
	{
		var a,t,i;

		a = domElm.getElementsByTagName("A");

		if( a[0] )
		{
			for(i = 0, t = a.length; i < t; i++)
			{
				pst.removeEvent(a[i],"focus",pst.keyboard.focus);
				pst.removeEvent(a[i],"blur",pst.keyboard.blur);
			}
		}

		pst.removeEvent(domElm,"keydown",pst.keyboard.nav);

	},

	setNav: function( domElm )
	{
		var a,t,i;

		a = domElm.getElementsByTagName("A");

		if( a[0] )
		{
			for(i = 0, t = a.length; i < t; i++)
			{
				pst.addEvent(a[i],"focus",pst.keyboard.focus);
				pst.addEvent(a[i],"blur",pst.keyboard.blur);

				if( i == a.length-1 )
				{
					pst.navKeyLastLink = a[i];
				}
			}
			pst.keyboard.setFocus( a[0] );

			pst.addEvent( domElm , "keydown", pst.keyboard.nav );
		}
	},

	nav: function(event)
	{
		pst.captureEvent(event);

		var i,n,p,a,f,e,x,v,t;

		n 	= ( keyCode == 39 || keyCode == 40 ) ? true : false;
		p 	= ( keyCode == 38 || keyCode == 37 ) ? true : false;

		if( n || p )
		{
			a = this.getElementsByTagName("A");
			f = 0;
			e = a.length;

			for(i = 0; i < e; i++ )
			{
				if( a[i].isFocus === true )
				{
					x = i+1;
					x = (x > e-1 ) ? e-1 : x;
					v = i-1;
					v = (v < 0) ? 0 : v;
					t = (n) ? a[x] : a[v];

					pst.keyboard.setFocus(t);

					break;
				}
			}
			return false;
		}
	},

	blur: function()
	{
		if( keyCode == 9 && this == pst.navKeyLastLink )
		{
			setTimeout(function(){
				pst.navKeyLastLink.focus();
			},10);
		}
	},

	focus: function()
	{
		pst.keyboard.setFocus(this);
	},

	setFocus: function( newLink )
	{
		if( newLink )
		{
			if( pst.currentFocusLink )
			{
				pst.currentFocusLink.isFocus 	= false;
			}
			pst.currentFocusLink 			= newLink;
			pst.currentFocusLink.isFocus 	= true;
			pst.currentFocusLink.focus();
		}
	}
};

// +++++++++++++++++++++++++++++++++++++++++++++
// PLUGINS - ACCESSIBILITY
// +++++++++++++++++++++++++++++++++++++++++++++

pst.accessibility = {

	fontSet: ["50","62.5","80","120","140","160"],

	init: function()
	{
		if(!document.getElementById){return;}

		var accessibilityTools = document.getElementById("euAccessibilityTools");

		if( accessibilityTools )
		{
			pst.accessibility.initFontSize();

			pst.addEvent(accessibilityTools,"click",pst.accessibility.captureTools);
			pst.addEvent(document,"keydown",pst.accessibility.captureTools);

			accessibilityTools.style.display = "block";
		}
	},

	captureTools: function(event)
	{
		pst.captureEvent(event);

		if( event.type == "keydown" )
		{
			if( /TEXTAREA|INPUT|SELECT/i.test( euElement.tagName ) )
			{
				return true;
			}
			else if( keyCode == 107 ) // +
			{
				pst.accessibility.increaseFontSize();
				return true;
			}
			else if( keyCode == 109 ) // -
			{
				pst.accessibility.decreaseFontSize();
				return true;
			}
			else if( keyCode == 13 ) // ENTER
			{
				euElement = euElement.getElementsByTagName("IMG");

				if( !euElement[0] )
				{
					return;
				}
				else
				{
					euElement = euElement[0];
				}
			}
			else
			{
				return;
			}
		}
		
		if( euElement.className == "noCss" )
		{
			euElement = euElement.parentNode.getElementsByTagName("IMG")[0];
		}
		
		if( euElement.className == "printPage" )
		{
			print();
			return false;
		}
		else if( euElement.className == "increaseText" )
		{
			pst.accessibility.increaseFontSize();
			return false;
		}
		else if( euElement.className == "decreaseText" )
		{
			pst.accessibility.decreaseFontSize();
			return false;
		}
	},

	initFontSize: function()
	{
		cookieFontSize	= pst.getCook("fontSize");

		if(!cookieFontSize || cookieFontSize > 4 || cookieFontSize < 0 || isNaN(cookieFontSize) )
		{
			cookieFontSize = 1;
		}

		pst.accessibility.applyFontSize(cookieFontSize);

	},

	applyFontSize: function( cookieFontSize )
	{
		if(!document.getElementsByTagName){return;}

		pst.setCook("fontSize",cookieFontSize);

		document.getElementsByTagName("BODY")[0].style.fontSize =	this.fontSet[cookieFontSize] + "%";

	},

	increaseFontSize: function()
	{
		cookieFontSize++;

		if( cookieFontSize >= this.fontSet.length )
		{
			cookieFontSize = this.fontSet.length;
		}
		pst.accessibility.applyFontSize(cookieFontSize);
	},

	decreaseFontSize: function()
	{
		cookieFontSize--;

		if( cookieFontSize <= 0 )
		{
			cookieFontSize = 0;
		}
		pst.accessibility.applyFontSize(cookieFontSize);
	}
};

// +++++++++++++++++++++++++++++++++++++++++++++
// PLUGINS - WEBSERVICE
// +++++++++++++++++++++++++++++++++++++++++++++


pst.webservice = {

	init: function()
	{
		pst.addEvent(document,"click",pst.webservice.captureClickOrKeypress);
		pst.addEvent(document,"keypress",pst.webservice.captureClickOrKeypress);
	},

    captureClickOrKeypress: function(event)
    {
    	pst.captureEvent(event);
		
		if( keyCode == 13 && euElement.tagName == 'A' && euElement.className.indexOf('euWsLink') != -1  )
		{
			pst.lastFocus 		= euElement;
			document.keyCoord 	= pst.getPosition(euElement);
			euElement 			= euElement.getElementsByTagName("IMG")[0];
		}
		else if( event.button != 0 )
		{
			return;
		}

		if( euElement.tagName == 'IMG' )
		{
	    	if( /euWsIco|euWsRetry/.test(euElement.className) )
			{
				pst.lastFocus = euElement.parentNode;

				pst.webservice.showPopup( euElement );

				return false;
			}
			else if( /euWsLoading|euWsWarning/.test(euElement.className) )
			{
				pst.lastFocus = euElement.parentNode;
				return false;
			}
		}
    },

	showPopup: function( srcElm )
	{
		// ---------------------------------------------
		// INIT LOADING
		// ---------------------------------------------

		srcElm.className	= "euWsLoading";
		srcElm.title		= pst.translate.label("wsLoading");

		// ---------------------------------------------
		// GET WEBSERVICE URL
		// ---------------------------------------------

		var urlWebservice	= srcElm.parentNode.getElementsByTagName("S");
			urlWebservice 	= (urlWebservice[0]) ? urlWebservice[0].innerHTML : '';

		// ---------------------------------------------
		// STORE SOURCE ELEMENT
		// ---------------------------------------------

		pst.currentWS		= srcElm;

		// ---------------------------------------------
		// NO CROSS-DOMAINE !!!!
		// ---------------------------------------------

		if( urlWebservice.indexOf("http://") === 0 )
		{
			pst.currentWS.className = "euWsWarning";
			pst.currentWS.title 	= pst.translate.label("wsDenied");

			pst.popup.show({
				content	: pst.translate.label("wsDenied"),
				pClass	: "euPopupError",
				pWidth	: 155
			});

			return false;
		}

		// ---------------------------------------------
		// START CALLING THE WEBSERVICE
		// ---------------------------------------------

		//urlWebservice = "/wel/template-2009/scripts/language_all.xml";

		pst.load({

			url		: urlWebservice.replace(/&amp;/ig,"&"),
			success	: pst.webservice.success,
			error	: pst.webservice.error

		});

	},

	success: function(txt,xml,cfg)
	{
		var xmlToHTML 	= pst.webservice.getList(xml);
		var cntList		= xmlToHTML.lst;
		var nbrError 	= xmlToHTML.nbr;
		var nbrTranslate= xmlToHTML.cnt;

		if( cntList !== '' )
		{
			pst.currentWS.className	= "euWsIco";
			pst.currentWS.title		= pst.translate.label("wsSuccess");

			//cntList = "<h5>" + pst.translate.label("wsTitle") + "</h5>" + cntList;

			pst.popup.show({
				content	: cntList,
				pClass	: "euPopup",
				pWidth	: 220 //175
			});
		}
		else if( nbrError === 0 &&  nbrTranslate === 0 )
		{
			pst.currentWS.className = "euWsRetry";
			pst.currentWS.title		= pst.translate.label("wsRetry");

			pst.popup.show({
				content	: pst.translate.label("wsRetry"),
				pClass	: "euPopup",
				pWidth	: 155
			});
		}
		else if( nbrError > 0 || cntList === "" )
		{
			pst.currentWS.className = "euWsWarning";
			pst.currentWS.title		= pst.translate.label("wsError");

			pst.popup.show({
				content	: pst.translate.label("wsError"),
				pClass	: "euPopupError",
				pWidth	: 155
			});
		}
	},

	error: function()
	{
		pst.currentWS.className = "euWsWarning";
		pst.currentWS.title		= pst.translate.label("wsError");

		pst.popup.show({
			content	: pst.translate.label("wsError"),
			pClass	: "euPopupError",
			pWidth	: 155
		});
	},

	getList: function(xml,mode) // CREATE LANGUAGE LIST OR ICON FROM XML
	{
		var s,t,o,p,b,r,a,i,j,e,z,n,l='';

		z = xml.getElementsByTagName("error");
		d = xml.getElementsByTagName("document");
		p = d.length;
		n = z.length;
		c = false;
		k = pst.translate.docLang;

		for( i = 0; i < p; i++ )
		{
			// ---------------------------------------------
			// GET/SET INFO FOR THIS LINK
			// ---------------------------------------------

			b = d[i];
			r = b.getAttribute("lang");
			a = b.getAttribute("label");
			t = b.getAttribute("type");
			t = (t=="unofficial") ? " unofficial" : "";
			e = b.getAttribute("href");
			s = ( k == r ) ? "selected" : "";

			if( r !== "" )
			{
				// ---------------------------------------------
				// BUILD THE LIST FOR THE DROPDOW MENU
				// ---------------------------------------------

				if( mode == "list" )
				{
					if( s == "selected" )
					{
						l += '<li class="currentLang"><a href="javascript:void(0)" lang="'+r+'">'+a+'</a></li>';
					}
					else if( t !== "" && c === false ) // first unofficial
					{
						l += "<li class='langSeparate'><a href='"+e+"' lang='"+r+"' hreflang='"+r+"' class='"+t+"'>"+a+"</a></li>";
						c = true;
					}
					else // official
					{
						l += "<li><a href='"+e+"' lang='"+r+"' hreflang='"+r+"' class='"+t+"'>"+a+"</a></li>";
					}
				}

				// ---------------------------------------------
				// BUILD THE LIST OF "ICO" LANGUAGE
				// ---------------------------------------------

				else
				{
					l+= '<a class="euLang" href="'+e+'" lang="'+r+'" title="'+a+'">'+r+'</a> ';
				}
			}
		}
		return {lst:l,nbr:n,cnt:p};
	}
};

// +++++++++++++++++++++++++++++++++++++++++++++
// PLUGIN - MENU
// +++++++++++++++++++++++++++++++++++++++++++++

pst.menu = {

	init: function() // FILL MENU FROM NOSCRIPT
	{
		if(!document.getElementById){return;}

		pst.menu.box = document.getElementById("euLanguageNoScript");

		if( pst.menu.box )
		{
			var langNoScriptA 	= pst.menu.box.getElementsByTagName("a");

			if( langNoScriptA )
			{
				var langLength 	= langNoScriptA.length;

				if( langLength > 0 )
				{
					var langNewList = "";
					var e;

					for( var i = 0; i < langLength; i++ )
					{
						e = langNoScriptA[i];

						if( i === 0 )
						{
							pst.menu.box.longName = e.title;
							langNewList += "<li class='"+e.className+"'><a href='javascript:void(0)' lang='"+e.lang+"' hreflang='"+e.lang+"' class='"+e.className+"'>"+e.title+"</a></li>";
						}
						else
						{
							langNewList += "<li><a title='"+e.title+"' href='"+e.href+"'>"+e.title+"</a></li>";
						}
					}

					if( langNewList !== '' )
					{
						pst.menu.box.id			= "euLangsForm";
						pst.menu.box.innerHTML 	= "<ul>"+langNewList+"</ul>";

						var service = document.getElementById("euService");

						if( service )
						{
							service.style.paddingRight = "150px";
						}

						if( pst.menu.box.className == "webservice" )
						{
							pst.menu.box.webService = 1;
						}

						// BIND EVENT
						pst.addEvent(pst.menu.box,"click",pst.menu.captureEvent);
					}
				}

			}

		}
	},

	out: function() // ON MOUSE OUT
	{
		pst.menu.timer = setTimeout(function(){pst.menu.toggle();},1000);
	},

	over: function() // ON MOUSE OVER
	{
		clearTimeout(pst.menu.timer);
	},

	captureEvent: function(event) // CAPTURE EVENT ON MENU
	{
		pst.captureEvent(event);

		if( event.type == "keydown" )
		{
			if( keyCode == 27 )
			{
				pst.menu.toggle();
				return;
			}
		}
		else if( event.type == "click" )
		{
			if( !euElement.className )
			{
				return true;
			}
			else if( pst.menu.box.webService == 1 )
			{
				pst.menu.box.id 		= "euLangsLoading";
				pst.menu.box.oldTitle 	= pst.menu.box.title; // Store label

				setTimeout(function(){
					pst.load({
						url 	: '/cgi-bin/coverage/coverage?url='+location.href,
						//url: "/wel/template-2009/scripts/language_all.xml?rand="+pst.rand(111,999),
						success	: function(txt,xml,cfg){pst.menu.fill(xml);},
						error	: pst.menu.toggle // display the non script version
					});
				},250);

				pst.menu.box.webService = 2;

				return false;
			}
			else if( /currentLink|selected/i.test(euElement.className) )
			{
				pst.menu.toggle();
				return false;
			}
		}

	},

	fill: function(xml) // FILL MENU FROM WEBSERVICE XML
	{
		var l = pst.webservice.getList(xml,"list");
		var c = l.lst;
		var y = l.cnt;

		if( y > 0 )
		{
			c = "<li class='selected'><a class='currentLink' href='javascript:void(0)'>"+pst.menu.box.longName+"</a></li>" + c;
			pst.menu.box.innerHTML = "<ul>"+c+"</ul>";
		}

		pst.menu.toggle();
	},

	toggle: function() // TOGGLE MENU
	{
		if( /euLangsForm|euLangsLoading/i.test(pst.menu.box.id) )
		{
			pst.menu.box.id 		= "euLangsShow";
			pst.menu.box.innerHTML 	= pst.menu.box.innerHTML; // IE6 ?

			pst.addEvent(pst.menu.box,"mouseout",pst.menu.out);
			pst.addEvent(pst.menu.box,"mouseover",pst.menu.over);
			pst.addEvent(pst.menu.box,"keydown",pst.menu.captureEvent);

			pst.keyboard.setNav(pst.menu.box);

			pst.hideEmbed("hidden");
		}
		else
		{
			pst.menu.box.id 		= "euLangsForm";
			pst.menu.box.innerHTML 	= pst.menu.box.innerHTML; // IE6 ?

			pst.removeEvent(pst.menu.box,"mouseout",pst.menu.out);
			pst.removeEvent(pst.menu.box,"mouseover",pst.menu.over);
			pst.removeEvent(pst.menu.box,"keydown",pst.menu.captureEvent);

			pst.keyboard.unSetNav(pst.menu.box);

			pst.hideEmbed("visible");
		}

		pst.menu.box.title 	= pst.menu.box.oldTitle; // restore label

		pst.keyboard.setFocus( pst.menu.box.getElementsByTagName("a")[0] );

	}
};

// +++++++++++++++++++++++++++++++++++++++++++++
// PLUGIN - COMPONENTS
// +++++++++++++++++++++++++++++++++++++++++++++

pst.components = {

	list:{}, // used to store reference of multiple instance components objet

	// ---------------------------------------------
	// PLUGINS ARRAY CONFIG
	// ---------------------------------------------
	// "components" : className of components
	// p			: push to array for multiple instance object
	// l			: load js collection -> "nametolowercase,othername..."
	// f			: callback function
	// ---------------------------------------------

	config: {

		"euPagin" 				: {p:"euPagin",l:"eupagin"},
		"euSlideShowConfig" 	: {p:"euSlideShow",l:"euslideshow"},
		"euMediaGalleryConfig" 	: {p:"euMediaGallery",l:"eumediagallery"},
		"euSimpleCalendar" 		: {p:"euSimpleCalendar",l:"eucalendar"},
		"euDynCalendar" 		: {p:"euDynCalendar",l:"eucalendar",f:function(o){o.style.display="none";}},
		"euArchiveCalendar" 	: {p:"euArchiveCalendar",l:"eucalendar",f:function(o){o.style.display="block";}},
		"euNoScriptCalendar" 	: {p:"euNoScriptCalendar",f:function(o){o.style.display="none";}},
		"euDynamicPanel" 		: {p:"euDynamicPanel",l:"eudynamicpanel"},
		"euTabsNoScript"		: {p:"euTabs",l:"eutabs",f:function(o){o.style.display="none";}}
	},

	init: function()
	{
		pst.include(pst.path+"eumodalwindow.js");

		if(!document.getElementsByTagName){return;}

		var z,u,t;

		u = document.getElementsByTagName("DIV");
		t = u.length;

		for(z=0; z < t; z++)
		{
			pst.components.check(u[z]);
		}
		pst.components.load();
	},

	check:function(o)
	{
		var c,p,l,f,plg;

		c = o.className;

		if( c )
		{
			for(plg in this.config )
			{
				p = this.config[plg].p; // push to X array
				l = this.config[plg].l; // load JS
				f = this.config[plg].f; // launch function

				if( c.indexOf(plg) != -1 )
				{
					if( p )
					{
						if(typeof pst.components.list[p] !== "object" )
						{
							pst.components.list[p] = [];
						}
						if(typeof pst.components.list[p] === "object" )
						{
							pst.components.list[p].push(o);
						}
					}
					return;
				}
			}
		}
	},

	load: function()
	{
		var c,p,l,f,t,i,z,u,plg;

		for(plg in this.config )
		{
			p = this.config[plg].p;
			l = this.config[plg].l;
			f = this.config[plg].f;

			if( pst.components.list[p] )
			{
				if( l )
				{
					l=l.split(",");

					for(i = 0, t = l.length; i < t; i++)
					{
						pst.include(pst.path+""+l[i]+".js");
					}
				}

				if( typeof f === "function" )
				{
					for(z = 0, u = pst.components.list[p].length; z < u; z++)
					{
						f( pst.components.list[p][z] );
					}
				}
			}
		}
	}

};


// +++++++++++++++++++++++++++++++++++++++++++++
// PLUGIN - TRANSLATE
// +++++++++++++++++++++++++++++++++++++++++++++

pst.translate = {

	docLang : "en", // assume as the default language of the document

	init: function() // load the label file
	{
		pst.translate.getLang();
		pst.include(pst.path+"label.js");
	},

	// try to find the current language of the document

	getLang: function()
	{
		// --------------------------------------
		// FROM HTML
		// --------------------------------------
		
		var fromHTML = document.getElementsByTagName('HTML')[0];
		var haveLang = null;

		if( fromHTML )
		{
			haveLang = fromHTML.getAttribute("lang");

			if( haveLang )
			{
				haveLang = haveLang.split(",")[0];
				haveLang = haveLang.split("-")[0];
				pst.translate.docLang = haveLang;

				return;
			}
		}

		// --------------------------------------
		// FROM NOSCRIPT LANGUAGE
		// --------------------------------------

		var fromNoScriptPST	= document.getElementById('euLanguageNoScript');
		var fromNoScriptXSL	= document.getElementById('langsSelector');
		var fromNoScript	= (fromNoScriptPST) ? fromNoScriptPST : fromNoScriptXSL;

		if( fromNoScript )
		{
			var a 	= fromNoScript.getElementsByTagName("A");
			var curLangLink = null;

			for(var i = 0, len = a.length; i < len; i++)
			{
				curLangLink = a[i];

				if( /curlang|selected/ig.test(curLangLink.className) )
				{
					haveLang = 	curLangLink.getAttribute("lang");

					if( haveLang )
					{
						pst.translate.docLang = haveLang;
						break;
					}
				}
			}
		}
	},

	label: function(label) // return the label definition
	{
		if( pst.label )
		{
			if(!pst.label[pst.translate.docLang])
			{
				pst.translate.docLang = "en";
			}
			
			var currentLabel = pst.label[pst.translate.docLang][label];
		}
		return (currentLabel) ? currentLabel : "";
	}
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ON READY
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pst.ready(pst.run);

// ---------------------------------------------
// COMPONENTS - E-VOTING
// ---------------------------------------------
// Very bad script -> freeze browser onload !
// ---------------------------------------------
// l : language
// i : id of poll
// ---------------------------------------------

function euVoting(l,i)
{
	u  = "http://evoting.ec.europa.eu/vp/2.2/poll_distant.php?";
	u += "&"+"id_lg="+l;
	u += "&"+"poll_id="+i;
	u += "&"+"our_remote_host=http://evoting.ec.europa.eu";
	u += "&"+"main_action_url="+window.location.href;
	if(i===""){return;}
	document.write('<'+'script language="JavaScript" src="'+u+'">'+"<"+"/script"+">");
}