<!--
/* You may give each page an identifying name, server, and channel on
the next lines. */
var idx = document.title.indexOf("| ");
if(idx > 0 && idx <= 11) {
    s.pageName = document.location.pathname + " | " + document.title.substring(idx + 2);
}
s.server="";  //If your site has more than one domain serving the same content, the server variable may be used to track which of those domains visitors are using. The following JavaScript will populate the domain of the page into the server variable.
s.channel="";  //A channel, or site section, uses the channel variable to group similar traffic-related content, e.g. pages. You can use a channel to report visits, visitors, etc.
s.pageType="";  //The pageType variable is used only to designate a 404 Page Not Found Error Page. It only has one possible value, which is "errorPage."
s.prop14="";
s.prop20="";
s.prop21="";
/* Conversion Variables */
s.events="";
s.eVar1="";
s.eVar8="";
s.eVar12="";
/************* DO NOT ALTER ANYTHING BELOW THIS LINE ! **************/
var s_code=s.t();if(s_code)document.write(s_code)//-->