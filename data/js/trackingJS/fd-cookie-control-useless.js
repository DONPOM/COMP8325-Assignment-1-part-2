//<![CDATA[
			cookieControl({
				t: {
					title: '<p>This site uses cookies to store information on your computer.</p>',
					intro: '<p>Some of these cookies are essential to make our site work and others help us to improve by giving us some insight into how the site is being used.</p>',
					full:'<p>These cookies are set when you submit a form, login or interact with the site by doing something that goes beyond clicking some simple links.</p><p>We also use some non-essential cookies to anonymously track visitors or enhance your experience of this site. If you\'re not happy with this, we won\'t set these cookies but some nice features on the site may be unavailable.</p><p>To control third party cookies, you can also <a class="ccc-settings" href="browser-settings" target="_blank">adjust your browser settings.</a></p><p>By using our site you accept the terms of our <a href="http://fdstaging.angels.uk.com/pages/cookies">Privacy Policy</a>.</p>'
				},
				position:CookieControl.POS_LEFT,
				style:CookieControl.STYLE_TRIANGLE,
				theme:CookieControl.THEME_LIGHT, // light or dark
				startOpen:true,
				autoHide:5000,
				subdomains:true,
				protectedCookies: [], //list the cookies you do not want deleted, for example ['analytics', 'twitter']
				apiKey: '675dc50ee325ab2bf93ddbbc47f6a6ca4e384d3f', // Read from .cookie_control_api.tcl
				product: CookieControl.PROD_FREE,
				consentModel: CookieControl.MODEL_INFO,
				onAccept:function(){},
				onReady:function(){},
				onCookiesAllowed:function(){},
				onCookiesNotAllowed:function(){}
			});
			//]]>
		