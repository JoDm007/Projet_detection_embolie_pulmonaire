# backend/cli.py
# Administration locale, en ligne de commande (pas d'inscription libre dans l'interface).
#   python -m backend.cli create-user <identifiant> [--role admin|radiologue]
#   python -m backend.cli set-password <identifiant>
#   python -m backend.cli disable-user <identifiant>   /  enable-user <identifiant>
#   python -m backend.cli list-users
#   python -m backend.cli install-models [--model logistic_regression] [--fp-budget 4]
#   python -m backend.cli check-models
#   python -m backend.cli purge
#   python -m backend.cli audit [--limit 50]
import argparse
import getpass
import sys

from backend.core.config import get_settings
from backend.core.security import hash_password, password_problem
from backend.db.database import init_db
from backend.db.repositories import AuditRepo, UserRepo


def _ask_password() -> str:
    while True:
        pwd = getpass.getpass("Mot de passe : ")
        problem = password_problem(pwd)
        if problem:
            print(problem)
            continue
        if getpass.getpass("Confirmer : ") != pwd:
            print("Les deux saisies diffèrent.")
            continue
        return pwd


def _user_or_exit(username: str):
    user = UserRepo.get_by_username(username)
    if user is None:
        sys.exit(f"Utilisateur inconnu : {username}")
    return user


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m backend.cli", description="Administration PE-Seg AI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("create-user", help="créer un compte")
    p.add_argument("username")
    p.add_argument("--role", choices=["radiologue", "admin"], default="radiologue")
    sub.add_parser("set-password", help="changer un mot de passe").add_argument("username")
    sub.add_parser("disable-user", help="désactiver un compte").add_argument("username")
    sub.add_parser("enable-user", help="réactiver un compte").add_argument("username")
    sub.add_parser("list-users", help="lister les comptes")
    p = sub.add_parser("install-models", help="installer le modèle O4 dans app_data/models")
    p.add_argument("--model", default="logistic_regression")
    p.add_argument("--fp-budget", type=int, default=4)
    p.add_argument("--feature-spacing", choices=["unit", "dicom"], default="unit",
                   help="spacing utilisé pour les features O3 à l'entraînement (voir engines/classical.py)")
    sub.add_parser("check-models", help="vérifier les empreintes des modèles installés")
    sub.add_parser("purge", help="effacer maintenant les analyses expirées")
    p = sub.add_parser("audit", help="afficher le journal d'audit")
    p.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)

    settings = get_settings()
    init_db(settings.db_path)

    if args.cmd == "create-user":
        if UserRepo.get_by_username(args.username):
            sys.exit(f"L'utilisateur {args.username} existe déjà.")
        UserRepo.create(args.username, hash_password(_ask_password()), args.role)
        AuditRepo.log("compte_cree", "cli", args.username, f"role={args.role}")
        print(f"Compte {args.username} ({args.role}) créé.")

    elif args.cmd == "set-password":
        user = _user_or_exit(args.username)
        UserRepo.set_password(user["id"], hash_password(_ask_password()))
        AuditRepo.log("mot_de_passe_change", "cli", args.username)
        print("Mot de passe changé ; les sessions ouvertes sont fermées.")

    elif args.cmd in ("disable-user", "enable-user"):
        user = _user_or_exit(args.username)
        active = args.cmd == "enable-user"
        UserRepo.set_active(user["id"], active)
        AuditRepo.log("compte_active" if active else "compte_desactive", "cli", args.username)
        print(f"Compte {args.username} {'réactivé' if active else 'désactivé'}.")

    elif args.cmd == "list-users":
        for u in UserRepo.list():
            state = "actif" if u["active"] else "désactivé"
            print(f"{u['username']:<20} {u['role']:<11} {state:<10} créé le {u['created_at']}")

    elif args.cmd == "install-models":
        from backend.engines.registry import install_classical_models
        params = install_classical_models(settings.models_source_dir, settings.models_dir,
                                          args.model, args.fp_budget, args.feature_spacing)
        AuditRepo.log("modeles_installes", "cli", args.model, f"seuil={params['threshold']:.5f}")
        print(f"Modèle {args.model} installé dans {settings.models_dir / 'classical'}")
        print(f"  seuil @ {args.fp_budget} FP/patient : {params['threshold']:.5f}")
        print(f"  sensibilité patients (globale, CV) : {params.get('sens_patients_globale', 'n/a')}")
        if params["fold_assignments"]:
            print("  modèles par fold : présents (démo FUMPE honnête possible)")
        else:
            print("  ⚠ modèles par fold ABSENTS : les cas de démo FUMPE seront refusés.\n"
                  "    Relancer « python -m src.O4_classical_classifier.train_classifier » puis cette commande.")

    elif args.cmd == "check-models":
        from backend.engines.registry import CLASSICAL_DIR, verify_manifest
        problem = verify_manifest(settings.models_dir / CLASSICAL_DIR)
        print("Modèles classiques : OK" if problem is None else f"Modèles classiques : {problem}")

    elif args.cmd == "purge":
        from backend.services.retention import purge_expired
        print(f"{purge_expired(settings)} analyse(s) effacée(s).")

    elif args.cmd == "audit":
        for row in reversed(AuditRepo.last(args.limit)):
            print(f"{row['ts']}  {row['username'] or '-':<15} {row['action']:<22} "
                  f"{row['target'] or ''} {row['detail'] or ''}")


if __name__ == "__main__":
    main()
