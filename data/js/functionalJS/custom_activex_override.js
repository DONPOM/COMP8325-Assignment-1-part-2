// Please leave this comment in place
// Created by: Randy Drisgill (The Mossman)
// August 17, 2007
//
// THIS OVERRIDES THE OOTB SHAREPOINT FUNCTION WHICH CAUSES ACTIVEX INSTALL ISSUES
//
// Essentially it overrides the ootb sharepoint function which calls the activex object
// See http://support.microsoft.com/default.aspx/kb/931509 for info on the issue
//
// To use in masterpage use this syntax in the <head>:
//  	<asp:ContentPlaceHolder id="PlaceHolderAdditionalPageHead" runat="server"/>
//   	<script type="text/javascript" src="/_catalogs/masterpage/custom_activex_override.js"></script>


	function ProcessDefaultOnLoad(onLoadFunctionNames)
	{
		//** Uncomment this to see when this runs
		//alert('Fixing the Issue');
		
		ProcessPNGImages();
		UpdateAccessibilityUI();
		
		//** We comment out the offending ootb function
		//** and leave the rest of the functions as they were
		//ProcessImn();
		for (var i=0; i < onLoadFunctionNames.length; i++)
		{
			var expr="if(typeof("+onLoadFunctionNames[i]+")=='function'){"+onLoadFunctionNames[i]+"();}";
			eval(expr);
		}
		if (typeof(_spUseDefaultFocus)!="undefined")
			DefaultFocus();
	}	
