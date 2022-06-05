


 
   
   
   
 
 




 
window.glam_affiliate_id = '1655028855';
 

if ( !window.glamadapt_timer )
  window.glamadapt_timer = new Date().getTime();

window.GlamLogWithTimer = function(pLog)
{
  var glamadapt_elapsed = new Date().getTime() - window.glamadapt_timer;
  if (window.console != undefined )
    console.log(glamadapt_elapsed + ': ' + pLog);
}

window.glam_session = new Object();
window.glam_session.country_code = null;

if ( !window.glamadapt_pvid )
{
  window.glamadapt_pvid=7372141774197550732;
}


window.glam_session.country_code='AU';

window.glam_session.region_code='QLD';

function GlamGetGeoData(pName){
 var vName = (pName == 'cc' || pName == 'co' ? 'country_code' : pName);
 return (window.glam_session && window.glam_session[vName] ? window.glam_session[vName] : null);
}
function GlamGetSessionId(){
 return ( window.glam_session && window.glam_session.glam_sid ? window.glam_session.glam_sid : null );
}

function GlamGetReferrerUrl()
{
var glamParentIsPermitted = false;
var glamCurWin = window;
try {
 try {
  for (i = 0; i <= 10; i++) {
   if ((glamCurWin.parent != null) && (glamCurWin.parent != glamCurWin)) {
    var loc = glamCurWin.parent.location.toString();
    var x = loc.length;
    if (x > 0) {
     glamCurWin = glamCurWin.parent;
     glamParentIsPermitted = true;
    }
    else {
     glamParentIsPermitted = false;
     break;
    }
   }
   else {
    if (i == 0) { glamParentIsPermitted = true; }
     break;
   }
  }
 }
 catch (e)
 { glamParentIsPermitted = false; }

 if (glamCurWin.document.referrer.length == 0) {
  glamURL = glamCurWin.location.href;
 }
 else {
  if (glamParentIsPermitted) {
   glamURL = glamCurWin.location.href;
  }
  else {
   glamURL = glamCurWin.document.referrer;
  }
 }
}
catch (ex) {};
return glamURL;
}



 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 




if ( !window.glam_rtbf )
{
window.glam_rtbf = new Object();
window.glam_rtbf['s728x90'] = new Object();
window.glam_rtbf['s300x250'] = new Object();
window.glam_rtbf['s160x600'] = new Object();

window.glam_rtbf['s728x90']['pm'] = new Object();
window.glam_rtbf['s300x250']['pm'] = new Object();
window.glam_rtbf['s160x600']['pm'] = new Object();

window.glam_rtbf['s728x90']['pm'].pid = 's72890';
window.glam_rtbf['s300x250']['pm'].pid = 's300250';
window.glam_rtbf['s160x600']['pm'].pid = 's160600';

window.glam_rtbf['s728x90']['a9'] = new Object();
window.glam_rtbf['s300x250']['a9'] = new Object();
window.glam_rtbf['s160x600']['a9'] = new Object();

window.glam_rtbf['s728x90']['a9'].pid = 'g7x9p';
window.glam_rtbf['s300x250']['a9'].pid = 'g3x2p';
window.glam_rtbf['s160x600']['a9'].pid = 'g1x6p';
}

window.GlamRtbMaxCpm = function(slot)
{
 var rtb_providers = ['a9', 'pm'];
 var rtb_max_cpm = -1;
 for(rtbp in rtb_providers)
 {
    var rtb_provider=rtb_providers[rtbp];
    if ( window.glam_rtbf[slot] && window.glam_rtbf[slot][rtb_provider] && window.glam_rtbf[slot][rtb_provider].isAvailable && window.glam_rtbf[slot][rtb_provider].isAvailable() && window.glam_rtbf[slot][rtb_provider].getCpm && window.glam_rtbf[slot][rtb_provider].getCpm() > rtb_max_cpm )
    {
      rtb_max_cpm = window.glam_rtbf[slot][rtb_provider].getCpm();
    }
 }
 return rtb_max_cpm;
}

window.GlamRtbIsAvailable = function(slot)
{
 var rtb_providers = ['a9', 'pm'];
 for(rtbp in rtb_providers)
 {
    var rtb_provider=rtb_providers[rtbp];
    if ( window.glam_rtbf[slot] && window.glam_rtbf[slot][rtb_provider] && window.glam_rtbf[slot][rtb_provider].isAvailable && window.glam_rtbf[slot][rtb_provider].isAvailable())
    {
      return true;
    }
 }
 return false;
}


window.GlamShowRtbIfAvailable = function(slot)
{
 var rtb_providers = ['a9', 'pm'];
 for(rtbp in rtb_providers)
 {
    var rtb_provider=rtb_providers[rtbp];
    if ( window.glam_rtbf[slot] && window.glam_rtbf[slot][rtb_provider] && window.glam_rtbf[slot][rtb_provider].showIfAvailable && window.glam_rtbf[slot][rtb_provider].showIfAvailable())
    {
      return true;
    }
 }
 return false;
}

window.GlamProcessRtbSlot = function(slot, provider)
{
if ( window.glam_rtbf[slot] && window.glam_rtbf[slot][provider] && window.glam_rtbf[slot][provider].processRtbResults)
{
 return window.glam_rtbf[slot][provider].processRtbResults();
}
return false;
}
window.GlamCreateRtbSlot = function(slot, provider)
{
if ( !window.glam_rtbf[slot] )
 window.glam_rtbf[slot] = new Object();
if ( !window.glam_rtbf[slot][provider] )
 window.glam_rtbf[slot][provider] = new Object();
}
window.GlamSetRtbSlotAttribute = function(slot, provider, attr, value)
{
GlamCreateRtbSlot(slot, provider);
window.glam_rtbf[slot][provider][attr] = value;
}

 
window.glam_rtbf['s728x90'].floor = (parseFloat('')?parseFloat(''):parseFloat('.10'));
window.glam_rtbf['s300x250'].floor = (parseFloat('')?parseFloat(''):parseFloat('.10'));
window.glam_rtbf['s160x600'].floor = (parseFloat('')?parseFloat(''):parseFloat('.10'));
 

















function GlamAddEventListener(expression, call)
{
  if ( !window.glamadapt_event_listeners )
  {
    window.glamadapt_event_listeners = new Array();
  }
  event_listener = new Object();
  event_listener.expression = expression;
  event_listener.call = call;
  window.glamadapt_event_listeners.push(event_listener);
}


function GlamCreateIframeSlot(afid, adsize, slotname, zone, tile)
{
  var adslot = GlamCreateSlot(afid, adsize, slotname, zone, tile);
  if ( adslot && adslot.pos ) 
  {
   slotname = 'a'+adslot.pos;
   window.glamadapt_adslots[slotname].slot_tt = 'i';
  }
}

function GlamCreateSlot(afid, adsize, slotname, zone, tile)
{
  zone = ( zone ? zone : '/' );

  if ( !window.glamadapt_timer )
  {
    window.glamadapt_timer = new Date().getTime();
  }
  if ( !window.glamadapt_adslots )
  {
    window.glamadapt_adslots = {};
  }
  if ( !window.glamadapt_adscount )
  {
    window.glamadapt_adscount = 0;
  }
  if ( ! window.glamadapt_pvid )
  {
      window.glamadapt_pvid = Math.random()*10000000000000000;
  }
  if ( !window.glamadapt_dormant_created )
  {

      window.glamadapt_dormant_created = true;
      window.glamadapt_reskin_created = true;
      window.glam_reskin_called = true; // API v1 compat
      window.glam_dormant_called = true; // API v1 compat
      GlamCreateSlot(afid, "888x11", "dormant", zone, 999);

  }
/*
  if ( '6' != '0' && !window.glamadapt_dormant_created )
  {
      window.glamadapt_dormant_created = true;
      window.glamadapt_reskin_created = true;
      window.glam_reskin_called = true; // API v1 compat
      window.glam_dormant_called = true; // API v1 compat
      GlamCreateSlot(afid, "888x11", "dormant", zone, 999);
  }
*/
  if ( '' == '1' && !window.glamadapt_reskin_created )
  {
      window.glamadapt_reskin_created = true;
      window.glam_reskin_called = true; // API v1 compat
      window.glam_dormant_called = true; // API v1 compat
      GlamCreateSlot(afid, "888x11", "reskin", zone, 999);
  }
  if ( !tile )
  {
    window.glamadapt_adscount++;
  }
  adslotObject = new Object();
  adslotObject.afid = afid;
  adslotObject.ccalls = 0;
  adslotObject.acalls = 1;
  adslotObject.sz = adsize;
  adslotObject.slot = slotname;
  adslotObject.pos = ( tile ? tile : window.glamadapt_adscount );
  adslotObject.zone = ( zone ? zone : '/' );
  window.glamadapt_adslots["a" + adslotObject.pos] = adslotObject;
  GlamLogWithTimer('Added ' + slotname);
  return adslotObject;
}


function GlamTrack(afid)
{
  GlamGetContent(afid, true, true);
}

function GlamGetAds(afid, zone)
{
  GlamGetContent(afid, false, false, ( zone ? zone : '/' ));
}


function GlamGetContent(afid, async, track_only, zone)
{
  var reqsq = -9999;

  zone = ( zone ? zone : '/' );
  var glam_host = 'www35.glam.com';
  var ga_api_tt=(window==top ? 'i' : 'i');
  var ga_api_dt='nopda';
  var ga_track_inv=0;
  var ga_track_inv_mode='js';
  window.glam_affiliate_id = (window.glam_affiliate_id ? window.glam_affiliate_id : afid);

  var js_call = ('https:' == document.location.protocol ? 'https://s' : 'http://') +
                 (!track_only ? glam_host + '/gad/glamadapt_psrv.act' : 'www22.glam.com/gad/glamadapt_jsapi_track.act') +
                 '?;afid=' + afid + ';zone=' + ( zone.indexOf('/') > -1 ? zone : '/' + zone )+ ';ord=' + window.glamadapt_pvid + ';sz=1x1;ga_slot=yes;gszd=-;gsz=';
  var js_udata ='';
  var js_slotdata ='';
  for(slot in window.glamadapt_adslots)
  {
    if ( afid == window.glamadapt_adslots[slot].afid && zone == window.glamadapt_adslots[slot].zone )
    {
      js_call += window.glamadapt_adslots[slot].sz + ':' + window.glamadapt_adslots[slot].pos + '-';
    }
    if ( afid == window.glamadapt_adslots[slot].afid && zone == window.glamadapt_adslots[slot].zone && window.glamadapt_adslots[slot].udata )
    {
      js_udata += ';' + window.glamadapt_adslots[slot].udata;
    }
    if ( afid == window.glamadapt_adslots[slot].afid && zone == window.glamadapt_adslots[slot].zone && window.glamadapt_adslots[slot].slot_tt )
    {
      js_udata += ';slot.' + window.glamadapt_adslots[slot].pos + '.tt=' + window.glamadapt_adslots[slot].slot_tt;
    }
    if ( afid == window.glamadapt_adslots[slot].afid && zone == window.glamadapt_adslots[slot].zone )
    {
      js_slotdata += ';slot.' + window.glamadapt_adslots[slot].pos + '.name=' + escape(window.glamadapt_adslots[slot].slot);
    }
  }
   var ga_cli_time = new Date().getTime();

  js_call += js_slotdata + js_udata + ';ga_api_dt=' + ga_api_dt + ';ga_api_tt=' + ga_api_tt + ';tt=' + ga_api_tt + ';ga_cli_time=' + ga_cli_time + ';';
  js_call += ( reqsq>-999 ? ';reqsq=' + reqsq + ';' : '' );
  js_call += (window.glam && window.glam.SimpleMetrics ? ';ga_lib_sm=1;': '');
  js_call += (window.glam_hawkeye_called && window.glam_hawkeye_kvs ? ';' + window.glam_hawkeye_kvs + ';': '');
  js_call += (window.glam_pubmatic_called && window.glam_pubmatic_kvs ? ';' + window.glam_pubmatic_kvs + ';': '');
  js_call += ';new_sid=112340141774197546611;ga_srv_log=0;ga_cli_log=1;ga_do_cli_log=yes;ga_api=iframe;_g_cv=2;';

  if ( window != window.top )
  {
    var glam_url = GlamGetReferrerUrl();
    glam_url = glam_url.substr(0, 512);
    js_call += ';&ga_url=' + escape(glam_url.replace('#', '')) + '&;'
  }
  if ( track_only && ga_track_inv_mode == 'iframe' && document.getElementById("GLAM_JSAPI_TRACK"))
  {
    var glam_url = GlamGetReferrerUrl();
    glam_url = glam_url.substr(0, 512);
    js_call += ';ga_track_inv_mode=iframe;&ga_url=' + escape(glam_url.replace('#', '')) + '&;'
    document.getElementById("GLAM_JSAPI_TRACK").innerHTML='<iframe src="'+js_call+'" height="0" width="0" style="visibility:hidden;display:none"></iframe>';
  }
  else if ( async )
  {
   (function() {
   var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
   ga.src = js_call;
   (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(ga);
   })();
  }
  else 
  {
    document.write('<scr' + 'ipt type="text/javascript" language="JavaScript" src="' +
                   js_call  +
                   '"><' + '/sc' + 'ript>');
  }

  GlamLogWithTimer('Called GlamGetAds()' );
}

function GlamGetAdsDone()
{
  GlamLogWithTimer('GlamGetAds() done' );
  window.glamadapt_done = true;
  document.write('<scri'+'pt type="text/javascript">GlamDoGetAdsDone();</scri'+'pt>');

}



function GlamIsAvailable(slotname)
{
  GlamLogWithTimer('GlamIsAdAvailable() called');
  var adslot = GlamGetSlotObject(slotname);
GlamLogWithTimer('GlamIsAvailable ' + slotname);
  if ( adslot && adslot.show && adslot.he_slot && window.GlamIsHawkeyeAvailable && (GlamIsHawkeyeAvailable(adslot.he_slot) || adslot.he_shown) )
    return true;
  else if ( adslot && adslot.show && adslot.rtb_slot && window.GlamRtbIsAvailable  && (GlamRtbIsAvailable (adslot.rtb_slot) || adslot.rtb_shown) )
    return true;
  else if ( !adslot || !adslot.show || adslot.noad || !window.glamadapt_done || adslot.nopda || adslot.processing || adslot.ccalls >= adslot.acalls )
    return false;
  else
  {
    adslot.ccalls++;
    return true;
  }
}

function GlamDoGetAdsDone()
{
  window.glamadapt_done = true;

  if ( window.glamadapt_event_listeners )
  {
    for(i in window.glamadapt_event_listeners)
    {
      if ( eval(window.glamadapt_event_listeners[i].expression) )
         eval(window.glamadapt_event_listeners[i].call);
    }
  }

}


function GlamSetSlotData(slotname, slotdata)
{
  for(slotid in window.glamadapt_adslots)
  {
    if ( window.glamadapt_adslots[slotid].slot == slotname )
    {
      window.glamadapt_adslots[slotid].udata = slotdata;
      return window.glamadapt_adslots[slotid];
    }
  }
  return null;
}


function GlamGetSlotObject(slotname, context)
{
  if ( !context )
     context=window;

  for(slotid in context.glamadapt_adslots)
  {
    if ( context.glamadapt_adslots[slotid].slot == slotname )
    {
      return context.glamadapt_adslots[slotid];
    }
  }
  return null;
}

function GlamRegisterSlotRtbApiResponse(pax, slot, response)
{
  GlamLogWithTimer('GlamRegisterSlotRtbApiResponse(' + pax + ',' + slot + ')');

  GlamSetAdSlotVar(slot, 'pax_' + pax, response);
}


function GlamSetAdSlotVar(slotname, advar, advalue)
{
  for(slotid in window.glamadapt_adslots)
  {
    if ( window.glamadapt_adslots[slotid].slot == slotname )
    {
      window.glamadapt_adslots[slotid][advar]=advalue;
      break;
    }
  }
}

function GlamShowIfAvailable(slotname)
{
  var adslot = GlamGetSlotObject(slotname);
  if ( adslot && adslot.showWithCallback )
    adslot.showWithCallback();
  else if ( adslot && adslot.show )
    GlamShow(slotname, adslot);
  return true;
}


function GlamShowFromIframe(slotname, context)
{
  if ( !context )
  {
    context = window;
    for (i=0; i <= 10; i++)
    {
      try
      {
        if (context.glamadapt_adslots )
          break;
      }
      catch(err) {}
      try{context = context.parent;} catch(err){}
    }
  }
  var adslot = GlamGetSlotObject(slotname, context);
  if ( adslot )
    try { eval('(' + adslot.show.toString() + ')()'); } catch(err) {}
}

function GlamLoad(slotname, doShow)
{
  return true;
}




function GlamShowIframeSlot(slotname, domid)
{
  var adslot;
  if ( (adslot=GlamGetSlotObject(slotname)) && document.getElementById(domid) && !(adslot.shown_count && adslot.group_id) && adslot.adid )
  {
    var shown_count = (adslot.shown_count ? adslot.shown_count : 0) + 1;
    var slotname = 'a'+adslot.pos;
    window.glamadapt_adslots[slotname].shown_count = shown_count;
    var show_url = ( shown_count > 1 && adslot.refresh_url ? adslot.refresh_url : adslot.url ) + ';ga_output=html;ord=' + window.glamadapt_pvid;
    document.getElementById(domid).innerHTML='<ifra' + 'me id="' + adslot.reqid + '" name="' + adslot.reqid + '" ' + 
                   'width="' + adslot.width + '" height="' + adslot.height + '" ' + 
                   'frameborder="0" marginheight="0" marginwidth="0" scrolling="no" allowTransparency="true" ' +
                   'src="' + show_url  + '" ' +
                   '><' + '/ifra' + 'me>';
  }

}


function GlamShow(slotname, adslotobj)
{
  if ( !window.glamadapt_reskin_shown )
  {
    window.glamadapt_reskin_shown = true;
  }
  if ( !window.glamadapt_dormant_shown )
  {
    window.glamadapt_dormant_shown = true;
    GlamShow('dormant');
    GlamShow('native');
  }
  
  var adslot = (adslotobj ? adslotobj : GlamGetSlotObject(slotname));

  if ( adslot && adslot.tt && adslot.tt == 'i' )
  {
    document.write('<ifra' + 'me id="' + adslot.reqid + '" name="' + adslot.reqid + '" ' + 
                   'width="' + adslot.width + '" height="' + adslot.height + '" ' + 
                   'frameborder="0" marginheight="0" marginwidth="0" scrolling="no" allowTransparency="true" ' +
                   'src="' + adslot.url  + ';ga_output=html;" ' +
                   '><' + '/ifra' + 'me>');
    return true;
  }
  else
  return ( adslot && adslot.show ? adslot.show() : false );
}


function GlamShowFromParentIframe(name){
 if ( window.top.glamadapt_adslots ) {
  window.glamadapt_adslots=window.top.glamadapt_adslots;
  var adslot = GlamGetSlotObject(slotname);
  if ( adslot && adslot.show ) {
   slotname = 'a'+adslot.pos;
   window.glamadapt_adslots[slotname].show_string = window.glamadapt_adslots[slotname].show.toString();
   eval("window.glamadapt_adslots[slotname].show= " + window.glamadapt_adslots[slotname].show_string);
   return window.glamadapt_adslots[slotname].show();
  }
 }
 return false;
}

function GlamGetRequestVar(name){
 if(name=(new RegExp('[?&]'+encodeURIComponent(name)+'=([^&]*)')).exec(location.search))
  return decodeURIComponent(name[1]);
}



