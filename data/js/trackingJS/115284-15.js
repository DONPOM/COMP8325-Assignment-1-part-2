rubicon_cb = Math.random();
rubicon_rurl = document.referrer;
if (top.location == document.location) {
    rubicon_rurl = document.location;
}
rubicon_rurl = escape(rubicon_rurl);
window.rubicon_ad = "3565386" + "." + "js";
window.rubicon_creative = "3814658" + "." + "js";
rubicon_tag_code = "<a href=\"http://optimized-by.rubiconproject.com/t/11256/28578/115284-15.3565386.3814658?url=http%3A%2F%2Fpubads.g.doubleclick.net%2Fgampad%2Fclk%3Fid%3D67767567%26iu%3D%2F59666047%2Ftheguardian.com\" target=\"_blank\"><img src=\"http://assets.rubiconproject.com/campaigns/11256/23/25/82/1398765746campaign_file_prfk1i.gif\" border=\"0\" alt=\"\" /></a>";
rubicon_tag_code = rubicon_tag_code.replace(/##RUBICON_CB##/g, rubicon_cb);
document.write(rubicon_tag_code);
document.write("<script>\nvar _comscore = _comscore || [];\n(function() {\n var rp_cats = \"IAB3,IAB20,IAB12-1,IAB11-2,IAB1-1,IAB12-3,IAB1,IAB12,IAB3-7,IAB17-11,IAB11-1,IAB17-8,IAB12-2,IAB1-7,IAB11,IAB11-5,IAB11-4,IAB17\";\n var rp_cat = 24*1000;\n if (rp_cats) {\n	rp_cats = rp_cats.replace(/IAB/g,\"\").split(\",\");\n	if (rp_cats.length > 0 && rp_cats[0] && rp_cats[0].length > 0) {\n		rp_cat = rp_cats[0].split(\"-\")[0];\n		rp_cat = rp_cat * 1000;\n	}\n }\n var _comscore_switch = \"au\";\n _comscore.push({ c1: \"8\", c2: \"6135404\", c3: rp_cat, c4: \"28578\", c10: \"3814658\" }); \n if ( _comscore_switch == \"us\" )  {\n   (function() { var s = document.createElement(\"script\"), el = document.getElementsByTagName(\"script\")[0]; s.async = true; s.src = (document.location.protocol == \"https:\" ? \"https://sb\" : \"http://b\") + \".scorecardresearch.com/beacon.js\"; \n   el.parentNode.insertBefore(s, el); \n   })(); \n }\n   \n})();\n<\/script><DIV STYLE=\"height: 0px; width: 0px; overflow: hidden\">\n<script>\n(function() {var proto = document.location.protocol; var server=\"http://tap2-cdn.rubiconproject.com\"; if (proto == \"https:\") server=\"https://tap-secure.rubiconproject.com\"; document.write(\'<IFRAME SRC=\"\'+server+\'/partner/scripts/rubicon/emily.html?rtb_ext=1&pc=11256/28578&geo=as&co=au\" FRAMEBORDER=\"0\" MARGINWIDTH=\"0\" MARGINHEIGHT=\"0\" SCROLLING=\"NO\" WIDTH=\"0\" HEIGHT=\"0\" style=\"height: 0px; width: 0px\"></IFRAME>\'); })();\n<\/script>\n\n</DIV>");
document.write('<scr' + 'ipt type="text/javascript">'
+ 'RubiconAdServing.incBucketValue("session_count","15.28578",1,1417675419);'
+ 'RubiconAdServing.incBucketValue("visits_count","15.28578",1,1417675419);'
+ '</scr' + 'ipt>');
