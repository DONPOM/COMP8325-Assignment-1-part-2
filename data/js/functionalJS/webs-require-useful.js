
    require.config({
        waitSeconds: 60,
        paths: {
            sb3Branding: "cobrand/webs/sb3Branding",
            persist: "internal/sitebuilder/builderChrome/persistGlimpse"
        }
    });

    define("prefetchTranslationNamespaces",[],function(){
        return [
            "webs.bldr",
            "webs.module"
        ];
    });

    require(['internal/sitebuilder/builderChrome/builderChrome.app']);
