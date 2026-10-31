# backend/services/anonymisation.py
# Anonymisation des DICOM importés, avant toute lecture des pixels (cf. plan2 § 4.5).
# On efface les attributs qui identifient le patient, l'examen ou l'établissement, ainsi que
# tous les tags privés des constructeurs. Seule la géométrie utile à l'analyse est gardée.
from pathlib import Path

import pydicom
from pydicom.uid import generate_uid

# Attributs mis à blanc (mots-clés DICOM). Liste inspirée du profil de base PS3.15 annexe E,
# restreinte aux attributs effectivement rencontrés dans les angioscanners.
IDENTIFYING_KEYWORDS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime", "PatientSex",
    "PatientAge", "PatientWeight", "PatientSize", "PatientAddress", "PatientTelephoneNumbers",
    "OtherPatientIDs", "OtherPatientNames", "OtherPatientIDsSequence", "PatientMotherBirthName",
    "EthnicGroup", "PatientComments", "MilitaryRank", "MedicalRecordLocator",
    "AccessionNumber", "StudyID", "StudyDate", "StudyTime", "SeriesDate", "SeriesTime",
    "AcquisitionDate", "AcquisitionTime", "ContentDate", "ContentTime", "AcquisitionDateTime",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName", "StationName",
    "ReferringPhysicianName", "ReferringPhysicianAddress", "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy", "OperatorsName", "PhysiciansOfRecord",
    "RequestingPhysician", "RequestAttributesSequence", "DeviceSerialNumber",
    "StudyDescription", "SeriesDescription", "RequestedProcedureDescription",
    "PerformedProcedureStepDescription", "AdmittingDiagnosesDescription",
]
# Les UID permettent de relier l'examen au système de l'hôpital : ils sont remplacés par des
# UID neufs (et non supprimés, car le format de fichier DICOM les exige).
UID_KEYWORDS = ["StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID"]


class DicomRejected(Exception):
    """La série importée n'est pas analysable ; message destiné à l'utilisateur."""


def anonymise_directory(directory: Path) -> dict:
    """Anonymise en place tous les .dcm du dossier et contrôle qu'il s'agit d'une seule série CT.
    Renvoie des informations non identifiantes (nombre de coupes)."""
    files = sorted(directory.glob("*.dcm"))
    if not files:
        raise DicomRejected("Aucun fichier DICOM (.dcm) trouvé dans l'archive.")

    series_uids, n_slices = set(), 0
    new_uids = {kw: generate_uid() for kw in UID_KEYWORDS}  # mêmes UID pour toute la série
    for path in files:
        try:
            ds = pydicom.dcmread(path)
        except Exception:
            raise DicomRejected(f"Fichier DICOM illisible : {path.name}.")
        modality = getattr(ds, "Modality", "")
        if modality and modality != "CT":
            raise DicomRejected(f"Modalité {modality} non prise en charge : un scanner (CT) est attendu.")
        if "PixelData" not in ds or "ImagePositionPatient" not in ds:
            raise DicomRejected(f"{path.name} n'est pas une coupe d'image exploitable.")
        series_uids.add(getattr(ds, "SeriesInstanceUID", None))

        for kw in IDENTIFYING_KEYWORDS:
            if kw in ds:
                delattr(ds, kw)
        for kw, uid in new_uids.items():
            if kw in ds:
                setattr(ds, kw, uid)
        sop_uid = generate_uid()
        ds.SOPInstanceUID = sop_uid
        if getattr(ds, "file_meta", None) is not None:
            ds.file_meta.MediaStorageSOPInstanceUID = sop_uid
        ds.remove_private_tags()
        ds.PatientIdentityRemoved = "YES"
        ds.save_as(path)
        n_slices += 1

    if len(series_uids - {None}) > 1:
        raise DicomRejected("L'archive contient plusieurs séries : importer une seule série d'angioscanner.")
    if n_slices < 20:
        raise DicomRejected(f"Seulement {n_slices} coupes : une série d'angioscanner complète est attendue.")
    return {"n_slices": n_slices}
