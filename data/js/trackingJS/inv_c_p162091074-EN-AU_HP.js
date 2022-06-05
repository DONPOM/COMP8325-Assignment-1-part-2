/*
Copyright (c) 2013, comScore Inc. All rights reserved.
version: 5.0.3
*/
COMSCORE.SiteRecruit.Builder.config = {
    version: "5.0.3",

    // invitations' settings
    invitation: [

    {
        methodology: 2,
        projectId: 'p162091074',
        weight: 100,
        isRapidCookie: false,
        acceptUrl: 'http://survey2.surveysite.com/wix/p162091074.aspx',
        secureUrl: '',
        acceptParams: {
            raw: 'l=9',
            siteCode: '1061',
            cookie: [
            ],
            otherVariables: [
            ]
        },
        viewUrl: 'https://web.survey-poll.com/tc/CreateLog.aspx',
        viewParams: 'log=comscore/view/p162091074-view.log',
        content: '<table style="width:360px; padding:0px; margin:0px; border-collapse:collapse; border:1px solid #999999; background-color: #FFFFFF;"><tr><td>  <table style="border:0px; border-collapse:collapse; width: 100%;"><tr><td>  	<table style="border:0px; border-collapse:collapse; width: 100%;"><tr valign="top"><td>  		<img src="logo-stripe.gif" /></td><td><a href="Close" onclick="@declineHandler"><img border="0" src="close.gif" /></a></td></tr> <tr><td colspan="2"><img src="top-stripe.gif" /> 		</td></tr> 	</table>  	<table style="width:100%;"><tr><td>   			<div style="font-family: Verdana, Arial, Helvetica, sans-serif;	font-size: 11px; color: #000000; margin: 6px 6px 15px 4px;line-height:14px;">Microsoft is conducting an online survey to understand your opinions of the Microsoft.com Home Page.  If you choose to participate, the online survey will be presented to you when you leave the Microsoft.com Home Page. <br /><br /> Would you like to participate? </div>  <div align="center" style="margin: 0; padding: 0"> <input style="margin: 0; padding: 0" type="button" value="  Yes  " onclick="@acceptHandler" />&nbsp;&nbsp; <input style="margin: 0; padding: 0"  type="button" value=" No " onclick="@declineHandler" /> </div>  <div style="font-family: Verdana,Arial,Helvetica,sans-serif; font-size:11px; color:#000000; margin:15px 0px 6px 6px;"><a  style="color:#0000EE; text-decoration: underline; font-size:10px;"  href="http://privacy.microsoft.com/en-ca/default.mspx" target="_blank">Privacy Statement</a></div>   <img src="bottom-stripe.gif" /></td></tr></table> </td></tr></table> </td></tr></table>   ',
        height: 210,
        width: 390,
        revealDelay: 1000,
        horizontalAlignment: 1,
        verticalAlignment: 1,
        horizontalMargin: 0,
        verticalMargin: 0,
        isHideBlockingElements: false,
        isAutoCentering: true,
        url: 'SiteRecruit_Tracker.htm',
        trackerGracePeriod: 5
        ,
        usesSilverlight: false

        //silverlight config

        ,
        trackerWindow: {
            width: 400,
            height: 270,
            orientation: 1,
            offsetX: 0,
            offsetY: 0,
            hideToolBars: true,
            trackerPageConfigUrl: 'trackerConfig_p162091074-EN.js'
            // future feature: 
            //features: "location=0,menubar=0,resizable=1,scrollbars=1,toolbar=0"
        },
        Events: {
            beforeRender: function() {
},
            beforeDestroy: function() {
},
            beforeAccept: function() {
                var val = "";
                try {
                    val = WTOptimize.z2b20[0].zccdd.r_experimentID;
                } catch (e) {}
                COMSCORE.SiteRecruit.Builder.invitation.config.acceptParams.raw += '&wtExpId=' + val;
            },
            beforeDecline: function() {
},
            beforeRenderUpdate: function() {
},
            afterRender: function() {
}
        }
    }
    ]
};
COMSCORE.SiteRecruit.Builder.run();

