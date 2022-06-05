
	if (webs && webs.mode === "bldr" && webs.site && webs.site.hasSeo) {
		require(['apps/App', 'apps/Dropdown', 'apps/SeoBuilderApp']);
	} else {
		require(['apps/App', 'apps/Dropdown']);
	}