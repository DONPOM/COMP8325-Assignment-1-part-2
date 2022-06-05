
        var jreVersions = deployJava.getJREs();
        if (jreVersions.length == 0)
            pageTracker._setVar("Java: none");
        for (var i = 0; i < jreVersions.length; ++i)
            pageTracker._setVar("Java: " + jreVersions[i]);
    