
var mboxTags = tacProp.tags.split(";");
var kvvert_value = "";
var kvpagetype_value = "";
for (var i = 0; i < mboxTags.length; i++) {
	var tag = mboxTags[i].split("=");
	if(tag.length == 2) {
		switch(tag[0]) {
		case "kvvert" : kvvert_value = tag[1];
		break;
		case "kvpagetype" : kvpagetype_value = tag[1];
		break;
		}
	}
}
mboxCreate('HP_Global', 'Vert=' + kvvert_value, 'pageType=' + kvpagetype_value, 'path= ' +location.pathname);
