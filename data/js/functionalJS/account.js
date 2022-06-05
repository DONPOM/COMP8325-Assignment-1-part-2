/**
 * Creates a default new instance of the account login and regi modal
 *
 * <p><b>Require Path:</b> shared/account/instances/account</p>
 *
 * @module Shared
 * @submodule Shared.Account
 * @class AccountInstance
 * @static
**/
define('shared/account/instances/account',[
    'shared/account/views/login-modal',
    'shared/account/views/registration-modal',
    'foundation/models/user-data'
], function (LogInModal, RegistrationModal, userData) {
    // FIXME: Because components (sharetools) don't remove the login-trigger class
    // we need to disable auth components when the user is logged out. If the
    // userinfo.json is async and isLoggedIn becomes true later, the login modal has
    // functionality to destroy it.
    if (!userData.isLoggedIn()) {
        new LogInModal();
        new RegistrationModal();
    }
});
