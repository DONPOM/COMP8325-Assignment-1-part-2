
var language_js =
{
EmailAddressTooLong: "Email address is too long.",
ThankYou: "Thank You!",
EnterValidEmail: "Please enter a valid email address.",
YourDiscountDomainClub: "Your Discount Domain Club pricing is being displayed.",
YourAccountExecutiveis: "Your Account Executive is:",
Phone: "Phone:",
Ext: "Ext:",
Email: "Email:"
}
var pcUrlhost = window.location.host;
var urlKnownAliases = {};
urlKnownAliases[pcUrlhost+'/hosting/website-builder.aspx'] = [pcUrlhost+'/hosting/website-builder-new.aspx'];
urlKnownAliases[pcUrlhost+'/domains/searchbulk.aspx'] = [pcUrlhost+'/bulk-domain-search.aspx'];
urlKnownAliases[pcUrlhost+'/domains/domain-name-search.aspx'] = [pcUrlhost+'/domains/search-new2.aspx'];
var urlDisableDomainSearchTray = {};
urlDisableDomainSearchTray[pcUrlhost+'/domains/domain-name-search.aspx'] = true;
urlDisableDomainSearchTray[pcUrlhost+'/domains/searchbulk.aspx'] = true;
urlDisableDomainSearchTray[pcUrlhost+'/domains/domain-transfer.aspx'] = true;
urlDisableDomainSearchTray[pcUrlhost+'/tlds/gtld.aspx'] = true;
urlDisableDomainSearchTray[pcUrlhost+'/domainaddon/private-registration.aspx'] = true;
