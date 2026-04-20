const translations = {
  ar: {
    appName: "نظام مراقبة السلامة",
    langSwitch: "English",
    contact: "تواصل معنا",
    contactSub: "للدعم الفني تواصل معنا",

    navCameras: "الكاميرات",
    navDashboard: "لوحة التحكم",
    navLogout: "خروج",
    navLogin: "دخول",
    navRegister: "تسجيل",

    loginTitle: "تسجيل الدخول",
    loginSubtitle: "مرحباً بك في نظام المراقبة",
    empIdLabel: "رقم الموظف",
    empIdPlaceholder: "مثال: EMP-001",
    passwordLabel: "كلمة المرور",
    loginBtn: "دخول",
    noAccount: "ليس لديك حساب؟",
    registerLink: "سجّل الآن",

    registerTitle: "إنشاء حساب جديد",
    registerSubtitle: "أدخل بياناتك للتسجيل في النظام",
    nameLabel: "الاسم الكامل",
    namePlaceholder: "أدخل اسمك الكامل",
    deptLabel: "القسم",
    deptPlaceholder: "مثال: الإنتاج",
    confirmPassLabel: "تأكيد كلمة المرور",
    registerBtn: "إنشاء الحساب",
    hasAccount: "لديك حساب بالفعل؟",
    loginLink: "سجّل الدخول",

    cameraTitle: "المراقبة المباشرة",
    cameraSubtitle: "بث مباشر من كاميرات المصنع",
    liveLabel: "مباشر",
    noSignal: "لا توجد إشارة",
    detectedWorkers: "عمال مكتشفون",
    detectedViolations: "مخالفات",
    detectedCompliant: "ملتزمون",
    supervisorNote: "للإحصائيات والتنبيهات التفصيلية",
    goToDashboard: "اذهب للوحة التحكم",

    dashTitle: "لوحة التحكم",
    welcome: "مرحباً",
    statTotal: "إجمالي العمال",
    statActive: "نشط الآن",
    statViolations: "مخالفات اليوم",
    statCompliance: "نسبة الالتزام",

    workersTitle: "سجل العمال",
    searchPlaceholder: "ابحث بالاسم أو الرقم...",
    noWorkers: "لا يوجد عمال مسجلون",
    cardStatusActive: "نشط",
    cardStatusInactive: "غير نشط",

    alertsTitle: "آخر التنبيهات",
    noAlerts: "لا توجد تنبيهات حتى الآن",

    errIdRequired: "الرجاء إدخال رقم الموظف",
    errPasswordRequired: "الرجاء إدخال كلمة المرور",
    errNameRequired: "الرجاء إدخال الاسم الكامل",
    errDeptRequired: "الرجاء إدخال القسم",
    errPasswordMismatch: "كلمتا المرور غير متطابقتين",
    errInvalidCredentials: "رقم الموظف أو كلمة المرور غير صحيحة",
    errIdExists: "رقم الموظف مسجل مسبقاً",
    errNetwork: "خطأ في الاتصال بالخادم",
    successRegistered: "تم التسجيل بنجاح! يمكنك الدخول الآن",
    backToCameras: "العودة للكاميرات",
  },
  en: {
    appName: "Safety Monitoring System",
    langSwitch: "عربي",
    contact: "Contact Us",
    contactSub: "Contact us for technical support",

    navCameras: "Cameras",
    navDashboard: "Dashboard",
    navLogout: "Logout",
    navLogin: "Login",
    navRegister: "Register",

    loginTitle: "Login",
    loginSubtitle: "Welcome to the monitoring system",
    empIdLabel: "Employee ID",
    empIdPlaceholder: "e.g. EMP-001",
    passwordLabel: "Password",
    loginBtn: "Login",
    noAccount: "Don't have an account?",
    registerLink: "Register Now",

    registerTitle: "Create New Account",
    registerSubtitle: "Enter your details to register",
    nameLabel: "Full Name",
    namePlaceholder: "Enter your full name",
    deptLabel: "Department",
    deptPlaceholder: "e.g. Production",
    confirmPassLabel: "Confirm Password",
    registerBtn: "Create Account",
    hasAccount: "Already have an account?",
    loginLink: "Login",

    cameraTitle: "Live Monitoring",
    cameraSubtitle: "Live feed from factory cameras",
    liveLabel: "LIVE",
    noSignal: "No Signal",
    detectedWorkers: "Detected Workers",
    detectedViolations: "Violations",
    detectedCompliant: "Compliant",
    supervisorNote: "For detailed statistics and alerts",
    goToDashboard: "Go to Dashboard",

    dashTitle: "Dashboard",
    welcome: "Welcome",
    statTotal: "Total Workers",
    statActive: "Active Now",
    statViolations: "Today's Violations",
    statCompliance: "Compliance Rate",

    workersTitle: "Workers Registry",
    searchPlaceholder: "Search by name or ID...",
    noWorkers: "No registered workers",
    cardStatusActive: "Active",
    cardStatusInactive: "Inactive",

    alertsTitle: "Latest Alerts",
    noAlerts: "No alerts yet",

    errIdRequired: "Please enter Employee ID",
    errPasswordRequired: "Please enter password",
    errNameRequired: "Please enter full name",
    errDeptRequired: "Please enter department",
    errPasswordMismatch: "Passwords do not match",
    errInvalidCredentials: "Invalid Employee ID or password",
    errIdExists: "Employee ID already registered",
    errNetwork: "Server connection error",
    successRegistered: "Registration successful! You can now login",
    backToCameras: "Back to Cameras",
  }
};

let currentLang = localStorage.getItem('lang') || 'ar';

function t(key) {
  return translations[currentLang][key] || key;
}

function applyTranslations() {
  document.documentElement.lang = currentLang;
  document.documentElement.dir  = currentLang === 'ar' ? 'rtl' : 'ltr';
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
      el.placeholder = t(key);
    } else {
      el.textContent = t(key);
    }
  });
}

function toggleLang() {
  localStorage.setItem('lang', currentLang === 'ar' ? 'en' : 'ar');
  location.reload();
}

document.addEventListener('DOMContentLoaded', applyTranslations);
