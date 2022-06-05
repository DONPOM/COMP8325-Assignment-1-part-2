
<!--
// script to show new private message popup

if (confirm(PHP.unhtmlspecialchars("You have a new private message.\n\nSender: El Chorizo\nTitle: 'Welcome to Sniper Central - Sniper & Sharpshooter Forums'\n\nClick OK to view it, or cancel to hide this prompt.", true)))
{
	// Output when OK is clicked
	if (confirm("Open the message in a new window?\n\n(Press cancel to open in the current window.)"))
	{
		//
		var winobj = window.open(getBaseUrl() + "private.php?do=showpm&pmid=162088", "pmnew", "statusbar=yes,menubar=yes,scrollbars=yes,toolbar=yes,location=yes,directories=yes,resizable=yes,top=50,left=50");
		if (winobj == null)
		{
			alert("Unable to open a new browser window,\n This might be due to a 'popup blocker'");
		}
	}
	else
	{
		window.location = "private.php?do=showpm&pmid=162088";
	}
}
// end pm popup script
//-->
