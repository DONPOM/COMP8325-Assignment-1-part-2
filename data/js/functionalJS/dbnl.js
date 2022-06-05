// Huib Verweij feb 2004
// 1 - Functie om de noten naast de tekst te zetten +
// 2 - Functie om de regelnummers van genummerd proza aan en uit te zetten.
// 3 - Functie om de navigatieparameters te definieren (oude navbar.js)


//###########################################################################



// **********************************************************
// * HHV: functie om de div-heights van de notentabel       *
// * net zo groot te maken als die van de teksttabel        *
// **********************************************************
 

function linkImages()
{
        if (document.images)
        {

                for (i=0;i<document.images.length;i++)
                {
                        //arr_images=document.images[i].src.split("/");
                        //imageFile=arr_images[arr_images.length-1];
                        if ( document.images[i].alt=="illustratie" || document.images[i].alt=="titelpagina")
                        {
                                // make image clickable
                                document.images[i].alt="Toelichting";
                                document.images[i].onmouseover = function ()
                                {
                                        this.style.cursor="pointer";
                                };
                                document.images[i].onmouseout = function ()
                                {
                                        this.style.cursor="default";
                                };
                                document.images[i].onclick = function ()
                                {
                                        document.location.href='/overdbnl/beeldmateriaal.php';
                                };
                        }

                }
        }
}
function resize_notentabel() {
//	if ( document.getElementById("tekst-kolom") == null ) {
//		return;
//	}
	var tekst_kolom = document.getElementById("tekst-kolom")
	var noten_kolom = document.getElementById("noten-kolom")
        var teksten=new Array();
        var noten=new Array();

	if ( noten_kolom == null ) {
		return;
	}
	
        for ( var i = 0 ; i < noten_kolom.childNodes.length ; i++ )
        {
        	
                if ( noten_kolom.childNodes[i].className == "noten-blok" )
                {
                        noten.push(noten_kolom.childNodes[i]);
                }
        }
        var notenRechts = document.getElementsByName("right");
	var z=-1;
        for (var i = 0 ; i < noten.length; i++)
        {
                notenRechts[i+1].innerHTML=noten[i].innerHTML;
/*
		var hrefs=notenRechts[i+1].getElementsByTagName("a");
        	for (var j = 0 ; j < hrefs.length; j++)
			hrefs[j].style.zIndex=z--;
*/
        }	
}


//###########################################################################


// **********************************************************
// * HHV: functie om de regelnummers van genummerd proza    *
// * wel en niet te tonen                                   *
// **********************************************************
 


function genummerd_proza() {

	var gp;
	
	 // gebruik alleen-IE 'className' attribuut eerst
	gp = getSelectedElements( 'span', 'className', 'genummerd-proza' );

	if ( gp.length == 0 ) { // probeer 't met 't 'class' attribuut
  	    gp = getSelectedElements( 'span', 'class', 'genummerd-proza' );
	}
	
	if ( gp.length > 0 ) {
		var display = gp[0].style.display == "none" ? "inline" : "none";
		for ( var i = 0 ; i < gp.length ; i++ ) {
			gp[i].style.display = display;
		}
		// Probeer nu de tekst van de link te vervangen
		var a = document.getElementById("genummerd-proza-link");
		if ( display == "none" ) {
			a.firstChild.nodeValue = 'Regelnummers proza weergeven';
		}
		else {
			a.firstChild.nodeValue = 'Regelnummers proza laten vervallen';
		}
	}
}

function getSelectedElements( tagName, attrName, attrValue ) {
	var startSet;
	var endSet = new Array();
        if ( tagName ) {
		startSet = document.getElementsByTagName( tagName );
	}
	else {
		startSet = document.all ? document.all : 
				document.getElementsByTagName( "*" );
	}
	if ( attrName ) {
		for ( var i = 0 ; i < startSet.length ; i++ ) {
			if ( startSet[i].getAttribute( attrName )) {
				if ( attrValue ) {
					if ( startSet[i].getAttribute( attrName ) == attrValue ) {
						endSet[endSet.length] = startSet[i];
					}
				}
				else {
					endSet[endSet.length] = startSet[i];
				}
			}
		}
	}
	else {
		endSet = startSet;
	}
	return endSet;
}




//###########################################################################


/*
 * Nieuwe navbar.js waarin een heel simpel object gedefinieerd wordt
 * om de 'oude' "new NavBar()" en "navbar.setColors()" en "navbar.setSize()"
 * methoden op te vangen.
 * Huib Verweij, mei 2004.
 */

function navBarSetColors(
  border, hdrFg, hdrBg, hdrHiFg, hdrHiBg, itmFg, itmBg, itmHiFg, itmHiBg
) {

  this.border  = border;
  this.hdrFg   = hdrFg;
  this.hdrBg   = hdrBg;
  this.hdrHiFg = hdrHiFg;
  this.hdrHiBg = hdrHiBg;
  this.itmFg   = itmFg;
  this.itmBg   = itmBg;
  this.itmHiFg = itmHiFg;
  this.itmHiBg = itmHiBg;
}
function navBarSetSizes( border, padding, separator ) {

}

// constructor
function NavBar( width ) {
  this.width = width;
  this.setSizes = navBarSetSizes;
  this.setColors = navBarSetColors;
}



function onResize() {
}

function setNext(nextpage)
{
	if (nextpage!="#")
	{
	jQuery(".volgende").css({display:"inline"});
        next=nextpage;
	}
}

function setPrev(prevpage)
{
	jQuery(".vorige").css({display:"inline"});
        prev=prevpage;
}

var prev="";
var next="";
var current = "normaal";

function gopage(gotopage)
{
        if (gotopage!='')
        {
                document.location.href=gotopage;
        }
}

function setStyleSheet()
{
	if (current=="normaal")
	{
		current="groot";
	}
/*	else if (current=="groot")
		setActiveStyleSheet('klein');*/
	else
	{
		current="normaal";
	}
	setActiveStyleSheet(current);
		
}

function setActiveStyleSheet(title) 
{
	var i, a, main;
	for(i=0; (a = document.getElementsByTagName("link")[i]); i++) 
	{
		if(a.getAttribute("rel").indexOf("style") != -1
			&& a.getAttribute("title")) 
		{
			a.disabled = true;
			if(a.getAttribute("title") == title) 
				a.disabled = false;
		}
	}
}

 
function getActiveStyleSheet() {
 var i, a;
 for(i=0; (a = document.getElementsByTagName("link")[i]); i++) {
  if(a.getAttribute("rel").indexOf("style") != -1
  && a.getAttribute("title")
  && !a.disabled) return a.getAttribute("title");
  }
  return null;
}

function createCookie(name,value,days) {
  if (days) {
    var date = new Date();
    date.setTime(date.getTime()+(days*24*60*60*1000));
    var expires = "; expires="+date.toGMTString();
  }
  else expires = "";
  if (window.location.host.indexOf("dbnl.org")>-1)
	document.cookie = name+"="+value+expires+"; domain=.dbnl.org; path=/";
  else
	document.cookie = name+"="+value+expires+"; domain=.dbnl.nl; path=/";

}
function readCookie(name) {
  var nameEQ = name + "=";
  var ca = document.cookie.split(';');
  for(var i=0;i < ca.length;i++) {
    var c = ca[i];
    while (c.charAt(0)==' ') c = c.substring(1,c.length);
    if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
  }
  return null;
}
function getPreferredStyleSheet() {
  var i, a;
  for(i=0; (a = document.getElementsByTagName("link")[i]); i++) {
    if(a.getAttribute("rel").indexOf("style") != -1
       && a.getAttribute("rel").indexOf("alt") == -1
       && a.getAttribute("title")
       ) return a.getAttribute("title");
  }
  return null;
}

function init()
{
        if ( self.resize_notentabel )
        {
                resize_notentabel();
        }
/*
	var cookie = readCookie("style");
	var title = cookie ? cookie : getPreferredStyleSheet();
	//alert(title);
	current=title;
	if (title=="groot" || title=="klein" || title=="normaal")
		setActiveStyleSheet(title);
*/
	if (jQuery("#tekst_en_noten"))
	{
		//werkt in firefox
/*
		jQuery("#tekst_en_noten").css({height:"100%"});
		jQuery("#tekst_en_noten tr td").css({height:"1px"});
		jQuery("#tekst_en_noten tr:last td").css({height:"100%"});
*/
                // lelijke IE hack
                if (jQuery.browser.msie && (jQuery.browser.version.substr(0,1)=="6"||jQuery.browser.version.substr(0,1)=="7"))
                {
                        jQuery("td#left").next().css({height:"700px"});
                        jQuery("table#content").css({height:"700px"});
                }
                else
                        jQuery("td#left").next().css({height:"100%"});
                jQuery("#tekst_en_noten").after("<table id='tekst_en_noten' class='filler' cellspacing=0 cellpadding=0 style='height:100%'><tr><td id='text'>&nbsp;</td><td id='right'></td></tr></table>");
//		jQuery('#tekst_en_noten tr:last').after('<tr><td height="100%" id="text">&nbsp;</td><td></td></tr>');

	}
	if (jQuery("scansonly").html()!=null)
	{
//		jQuery('td').each(function () { if (this.id) this.id = this.id+"something" });
		jQuery("#text").css({background:"#E9E9EA"});
		jQuery(".text-bottom").css({background:"#E9E9EA"});
	}
	if (jQuery("a.media"))
	{
		jQuery('a.media').media({width:515, height:500});
//		jQuery('a.media').media();
	}
	if (jQuery("a.audio"))
	{
		jQuery('a.audio').media({width:220, height:24});
//		jQuery('a.media').media();
	}
	setSnelzoek();
	setTabs();

	if(jQuery("#video1").length>0)
	{
		jQuery('#video1').dialog({
			autoOpen: false,
			width: 500,
			height:500,
			position:"right"
		});
		jQuery("#video1").dialog('open');
	}
}
/* touch device functies */
function isTouchDevice(){
	try{
	document.createEvent("TouchEvent");
		return true;
	}catch(e){
		return false;
	}
}

function touchScroll(id)
{
	if(isTouchDevice())
	{ //if touch events exist...
		var el=document.getElementById(id);
		var scrollStartPos=0;

		document.getElementById(id).addEventListener("touchstart", function(event) {
			scrollStartPos=this.scrollTop+event.touches[0].pageY;
//			event.preventDefault();
		},false);

		document.getElementById(id).addEventListener("touchmove", function(event) {
			this.scrollTop=scrollStartPos-event.touches[0].pageY;
//			event.preventDefault();
		},false);
	}
}


window.onunload = function(e) {
	var title = getActiveStyleSheet();
	createCookie("style", title, 365);
}

function setSnelzoek()
{
	if (jQuery("#frm_keuze input[type='radio']:checked").val()=='titel')
	{
		jQuery('#snelzoek_auteurs').hide();
		jQuery('#snelzoek_titels').show();
		jQuery('#zoek_tekst').hide();
	}
	if (jQuery("#frm_keuze input[type='radio']:checked").val()=='auteur')
	{
		jQuery('#snelzoek_auteurs').show();
		jQuery('#snelzoek_titels').hide();
		jQuery('#zoek_tekst').hide();
	}
	if (jQuery("#frm_keuze input[type='radio']:checked").val()=='tekst')
	{
		jQuery('#snelzoek_titels').hide();
		jQuery('#snelzoek_auteurs').hide();
		jQuery('#zoek_tekst').show();
	}
		
}

function setTabs()
{
	jQuery('#tabs div').hide(); // Hide all divs
	var chosenTab=jQuery("#tabs ul li[class=active] a");
	if (chosenTab.html()!=null)
	{
		var c=chosenTab.attr('href');
		jQuery(c).show();
	}
	else
	{
		jQuery('#tabs div:first').show(); // Show the first div
		jQuery('#tabs ul li:first').addClass('active'); // Set the class of the first link to active
	}
	jQuery('#tabs ul li a').click(function(){ //When any link is clicked
		jQuery('#tabs ul li').removeClass('active'); // Remove active class from all links
		jQuery(this).parent().addClass('active'); //Set clicked link class to active
		var currentTab = jQuery(this).attr('href'); // Set variable currentTab to value of href attribute of clicked link
		jQuery('#tabs div.tabcontent').hide(); // Hide all divs
		jQuery(currentTab).show(); // Show div with id equal to variable currentTab
		return false;
	});
}

var state='none';

function showAuteur(layer_ref)
{
	jQuery('#zoek_tekst').hide();
	jQuery('#snelzoek_auteurs').show();
	jQuery('#snelzoek_titels').hide();
}
function showTitel(layer_ref)
{
	jQuery('#zoek_tekst').hide();
	jQuery('#snelzoek_auteurs').hide();
	jQuery('#snelzoek_titels').show();
}
function showTekst(layer_ref)
{
	jQuery('#zoek_tekst').show();
	jQuery('#snelzoek_auteurs').hide();
	jQuery('#snelzoek_titels').hide();
}

function showhide(layer_ref)
{
        if (state == 'block')
        {
                state = 'none';
        }
        else
        {
                state = 'block';
        }
        if (document.all)
        { //IS IE 4 or 5 (or 6 beta)
                eval( "document.all." + layer_ref + ".style.display = state");
        }
        if (document.layers)
        { //IS NETSCAPE 4 or below
                document.layers[layer_ref].display = state;
        }
        if (document.getElementById &&!document.all)
        {
                hza = document.getElementById(layer_ref);
                hza.style.display = state;
        }
}

function createPlayer(theFile) 
{
	createPlayerBig(theFile, "", true) 
}
function createPlayerBig(theFile, id, more) 
{
	theFile=getTheFile(theFile);
	var flashvars = {
		file:theFile,
		autostart:"false",
		'playlist.thumbs':'false',
		shuffle:"false",
		playlistsize:"150",
		playlist:"bottom",
		repeat:"list"
	}
	var playlist="bottom";
	var playlistsize="150";
	var heightval="455";

	if (!more)
	{
		var flashvars = {
			file:theFile,
			autostart:"false",
			shuffle:"false",
			playlistsize:"1",
			playlist:"none"
		}
		heightval="305";
		playlistsize="0";
		playlist="none";
	}
	var params = {
		allowfullscreen:"true",
		allowscriptaccess:"always"
	}
	var attributes = {
		id:"player1",
		name:"player1"
	}
//	swfobject.embedSWF("/js/jwplayer/player.swf", "container"+id, "500", height, "9.0.115", false, flashvars, params, attributes);
    //flashplayer: '/js/jwplayer/player.swf',
	jwplayer('container'+id).setup({
		levels: [
			{file: theFile}
		],
		autostart:"false",
		'playlist.thumbs':'false',
		shuffle:"false",
		playlistsize:playlistsize,
		playlist:playlist,
		repeat:"list",
		height: heightval,
		width: 500,
	'modes': [
        {type: 'flash', src: '/js/jwplayer/player.swf'},
        {
          type: 'html5',
          config: {
           'file': theFile,
           'provider': 'video'
          }
        }
    ]

	});
}

function createPlayerSmall(theFile, id) 
{
	theFile=getTheFile(theFile);
	var flashvars = {
		file:theFile,
		autostart:"false",
		shuffle:"false",
		playlistsize:"0",
		controlbar:"none",
		playlist:"none"
	}
	var params = {
		allowfullscreen:"true",
		allowscriptaccess:"always"
	}
	var attributes = {
		id:"player"+id,
		name:"player"+id
	}
//	swfobject.embedSWF("/js/jwplayer/player.swf", "container"+id, "181", "101", "9.0.115", false, flashvars, params, attributes);
   jwplayer('container'+id).setup({
    flashplayer: '/js/jwplayer/player.swf',
    levels: [
      {file: theFile}
    ],
    autostart:"false",
	controlbar:"none",
	playlist:"none",
    height: 101,
    width: 181
  });
}

function getTheFile(theFile)
{
	var hasFlash = false;
	try {
		var fo = new ActiveXObject('ShockwaveFlash.ShockwaveFlash');
		if(fo) hasFlash = true;
	}catch(e){
		if(navigator.mimeTypes ["application/x-shockwave-flash"] != undefined) hasFlash = true;
	}
	if (!hasFlash)
	{
		if (theFile.substring(0,7)=="/tekst/")
			theFile="/arch/video/"+theFile.substring(7,24)+".f4v";
		else		
			theFile="/arch/video/"+theFile.substring(0,17)+".f4v";
	}
	return theFile;
}

function showHiRes(img, text)
{
        var popup = window.open('','afbeelding','scrollbars=yes,resizable=yes,width=640,height=480');
        var html = '';
        html += '<html>';
        html += '<head><title>DBNL - Afbeelding</title>';
        html += '<scr'+'ipt>';
        html += 'var spaceX = 50; var spaceY = 80; var screenX = screen.availWidth;';
        html += 'var screenY = screen.availHeight; var image = new Image();';
        html += 'image.src = "'+img+'"; image.onload = function(){ imgX = this.width; imgY = this.height;';
        html += 'var x = spaceX + imgX; var y = spaceY + imgY; if (screenX<x) x = screenX-100;';
        html += 'if (screenY<y) y = screenY-100; window.resizeTo(x,y); window.moveTo(40,40); }';
        html += '</scr'+'ipt>';
        html += '<style type="text/css">body{font-family:Verdana,Arial,sans-serif; font-size:14px; line-height:18px;}</style>';
        html += '</head>';
        html += '<body>';
        html += '<a href="javascript:window.close();"><img border="0" src="'+img;
        html += '" alt="Klik op de afbeelding om dit venster te sluiten." /></a><br/>';
        html += text;
        html += '</body>';
        html += '</html>';
        popup.document.write(html);
        popup.document.close();
        popup.focus();
        //var popup = window.open('/zoom.php?thumb='+thumb+'&hires='+img+'&titel='+text,'afbeelding','scrollbars=yes,resizable=yes,width=1000,height=600');
        return false;
}
