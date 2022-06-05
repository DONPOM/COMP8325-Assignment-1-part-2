define([
	"instance/site/site",
	"site/appstore/AppList"
], function(site, AppList){
	var appList = new AppList();
	appList.url = site.url() + "/apps";
	return appList;
});
