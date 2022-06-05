

    function show(tab) {
        var intro = document.getElementsByName("intro");
        //var features = document.getElementsByName("features");
        var screenshots = document.getElementsByName("screenshots");
        var showIntro = tab == 'intro';
        //var showFeatures = tab == 'features';
        var showScreenshots = tab == 'screenshots';

        if (showIntro) {
            intro[intro.length - 1].style.display = "block";
        }
        else {
            intro[intro.length - 1].style.display = "none";
        }

        //        if (showFeatures) {
        //            features[features.length - 1].style.display = "block";
        //        }
        //        else {
        //            features[features.length - 1].style.display = "none";
        //        }

        if (showScreenshots) {
            screenshots[screenshots.length - 1].style.display = "block";
        }
        else {
            screenshots[screenshots.length - 1].style.display = "none";
        }
    }


    function showSlideShow(tab) {
        var iphoneW = document.getElementsByName("wrapperiPhone");
        var iphone = iphoneW[iphoneW.length - 1];
        var ipad = document.getElementsByName("wrapperiPad")[document.getElementsByName('wrapperiPad').length - 1];
        var android = document.getElementsByName("wrapperAndroid")[document.getElementsByName('wrapperAndroid').length - 1];
        var blackberry = document.getElementsByName("wrapperBlackberry")[document.getElementsByName('wrapperBlackberry').length - 1];
        var showIPhone = tab == 'iphone';
        var showIPad = tab == 'ipad';
        var showAndroid = tab == 'android';
        var showBlackberry = tab == 'blackberry';

        var logoIPad = document.getElementsByName('iPadLogo')[document.getElementsByName('iPadLogo').length - 1];
        var logoIPhone = document.getElementsByName('iPhoneLogo')[document.getElementsByName('iPhoneLogo').length - 1];
        var logoAndroid = document.getElementsByName('androidLogo')[document.getElementsByName('androidLogo').length - 1];
        var logoBlackberry = document.getElementsByName('blackberryLogo')[document.getElementsByName('blackberryLogo').length - 1];



        if (showIPhone) {
            iphone.style.display = "block";
            ipad.style.display = "none";
            android.style.display = "none";
            blackberry.style.display = "none";
            logoIPhone.src = "/App_Themes/Default/iphone/menupopup/logo_iPhone.png";
            logoIPad.src = "/App_Themes/Default/iphone/menupopup/logo_iPad_unsel.png";
            logoAndroid.src = "/App_Themes/Default/iphone/menupopup/logo_Android_unsel.png";
            logoBlackberry.src = "/App_Themes/Default/iphone/menupopup/logo_Blackberry_unsel.png";
        }
        else if (showIPad) {
            iphone.style.display = "none";
            ipad.style.display = "block";
            android.style.display = "none";
            blackberry.style.display = "none";
            logoIPhone.src = "/App_Themes/Default/iphone/menupopup/logo_iPhone_unsel.png";
            logoIPad.src = "/App_Themes/Default/iphone/menupopup/logo_iPad.png";
            logoAndroid.src = "/App_Themes/Default/iphone/menupopup/logo_Android_unsel.png";
            logoBlackberry.src = "/App_Themes/Default/iphone/menupopup/logo_Blackberry_unsel.png";
        }
        else if (showAndroid) {
            iphone.style.display = "none";
            ipad.style.display = "none";
            android.style.display = "block";
            blackberry.style.display = "none";
            logoIPhone.src = "/App_Themes/Default/iphone/menupopup/logo_iPhone_unsel.png";
            logoIPad.src = "/App_Themes/Default/iphone/menupopup/logo_iPad_unsel.png";
            logoAndroid.src = "/App_Themes/Default/iphone/menupopup/logo_Android.png";
            logoBlackberry.src = "/App_Themes/Default/iphone/menupopup/logo_Blackberry_unsel.png";
        }
        else if (showBlackberry) {
            iphone.style.display = "none";
            ipad.style.display = "none";
            android.style.display = "none";
            blackberry.style.display = "block";
            logoIPhone.src = "/App_Themes/Default/iphone/menupopup/logo_iPhone_unsel.png";
            logoIPad.src = "/App_Themes/Default/iphone/menupopup/logo_iPad_unsel.png";
            logoAndroid.src = "/App_Themes/Default/iphone/menupopup/logo_Android_unsel.png";
            logoBlackberry.src = "/App_Themes/Default/iphone/menupopup/logo_Blackberry.png";
        }
    }

