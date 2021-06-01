"""Job Efficiency from SlurmDB

"""

import collections
import datetime
import grp
import itertools
import pwd
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple
from operator import attrgetter
import os

import click
import matplotlib.pyplot as plt

# List of relevant sacct attributes
ATTRIBUTES = [
    "JobID",
    "UID",
    "GID",
    "Start",
    "End",
    "State",
    "AllocCPUS",
    "REQMEM",
    "TotalCPU",
    "AveCPU",
    "AveCPUFreq",
    "Elapsed",
    "MaxRSS",
    "ExitCode",
    "NNodes",
    "NTasks",
    "NodeList",
]

# sacct binary
SACCT = shutil.which("sacct")

# job record corresponding the the sacct attributes
JobRecord = NamedTuple(
    "JobRecord",
    [
        ("jobid", str),
        ("step", str),
        ("uid", int),
        ("gid", int),
        ("start", Optional[datetime.datetime]),
        ("end", Optional[datetime.datetime]),
        ("state", str),
        ("alloccpus", int),
        ("reqmem", str),
        ("totalcpu", datetime.timedelta),
        ("avecpu", datetime.timedelta),
        ("avecpufreq", Optional[int]),
        ("elapsed", datetime.timedelta),
        ("maxrss", Optional[int]),
        ("exitcode", int),
        ("exitsignal", int),
        ("nnodes", int),
        ("ntasks", Optional[int]),
        ("nodelist", List[str]),
    ],
)


@dataclass
class JobSummary:
    uid: int
    walltime: int
    cputime: int


# for click argument parsing of job ids
@dataclass
class JobIdsOption:

    value: str
    items: List[int]

    @staticmethod
    def parse(ctx, params, value):
        if value:
            try:
                items = [int(v) for v in value.split(",")]
            except ValueError as err:
                raise click.BadParameter(f"Bad value {err}", ctx)
        else:
            items = []

        return JobIdsOption(value, items)


# for click argument parsing of users
@dataclass
class UsersOption:

    value: str
    items: List[int]

    @staticmethod
    def parse(ctx, params, value):
        if value:
            users = value.split(",")
            try:
                items = [pwd.getpwnam(u).pw_uid for u in users]
            except KeyError as err:
                raise click.BadParameter(f"Bad value {err}", ctx)
        else:
            items = []

        return UsersOption(value, items)


# for click argument parsing of groups
@dataclass
class GroupsOption:

    value: str
    items: List[int]

    @staticmethod
    def parse(ctx, params, value):
        if value:
            groups = value.split(",")
            try:
                items = [grp.getgrnam(g).gr_gid for g in groups]
            except KeyError as err:
                raise click.BadParameter(f"Bad value {err}", ctx)
        else:
            items = []

        return GroupsOption(value, items)


@click.command()
@click.option("--jobids", "-j", callback=JobIdsOption.parse, default="")
@click.option("--users", "-u", callback=UsersOption.parse, default="")
@click.option("--groups", "-g", callback=GroupsOption.parse, default="")
@click.option("--until", type=click.DateTime(), default=datetime.datetime.now())
@click.option("--days", type=int, default=1)
def main(
    jobids: JobIdsOption,
    users: UsersOption,
    groups: GroupsOption,
    until: datetime.datetime,
    days: int,
):

    endtime = until.replace(microsecond=0)
    starttime = endtime - datetime.timedelta(days=days)
    job_records = query_slurmdb(
        jobids.items, users.items, groups.items, starttime, endtime
    )

    if starttime.year == endtime.year:
        title = (
            f"CPU Usage {starttime.strftime('%d.%m')} - {endtime.strftime('%d.%m.%y')}"
        )
    else:
        title = f"CPU Usage {starttime.strftime('%d.%m.%y')} - {endtime.strftime('%d.%m.%Y')}"

    jobeff_plot(job_records, title, "jobeff.png")
    scatter_plot(job_records, title, "jobeff_scatter.png")


def jobeff_plot(job_records: List[JobRecord], title: str, output: str):

    total = {}
    for job_id, job_steps in job_records.items():
        job = job_steps[""]
        walltime = job.alloccpus * job.elapsed
        if not job.totalcpu:
            continue
        cputime = job.totalcpu
        if job.uid in total:
            total[job.uid].walltime += walltime
            total[job.uid].cputime += cputime
        else:
            total[job.uid] = JobSummary(job.uid, walltime, cputime)

    names = []
    walltime = []
    cputime = []
    for js in sorted(total.values(), key=attrgetter("walltime")):
        names.append(initials(js.uid))
        walltime.append(js.walltime / datetime.timedelta(hours=1))
        cputime.append(js.cputime / datetime.timedelta(hours=1))

    fig, ax = plt.subplots()
    ax.barh(names, walltime, color="lavender", edgecolor="blue")
    ax.barh(names, cputime, color="blue")
    plt.ylabel("User")
    plt.xlabel("CPU Hours")
    plt.suptitle(title)

    plt.show()
    plt.savefig(output)


def scatter_plot(job_records: List[JobRecord], title: str, output: str):

    walltime = []
    cputime = []
    for job_id, job_steps in job_records.items():
        job = job_steps[""]
        if not job.totalcpu:
            continue
        walltime.append(job.alloccpus * job.elapsed / datetime.timedelta(hours=1))
        cputime.append(job.totalcpu / datetime.timedelta(hours=1))
        if cputime[-1] > walltime[-1]:
            print(
                f"{job_id} - {cputime[-1]/walltime[-1]*100:5.2f}% - {walltime[-1]:5.0f} hours"
            )

    fig, ax = plt.subplots()
    plt.scatter(walltime, cputime, s=10)
    plt.axline((0, 0), (10, 10), color="red")
    plt.axline((0, 0), (10, 7.5), color="orange")
    plt.axline((0, 0), (10, 20.0), color="yellow")
    plt.xlabel("NCPU * Time Elapsed [h]")
    plt.ylabel("CPUTime [h]")
    plt.suptitle(title)
    plt.show()
    plt.savefig(output)


def query_slurmdb(
    jobids: List[int],
    uids: List[int],
    gids: List[int],
    starttime: datetime.datetime,
    endtime: datetime.datetime,
) -> List[JobRecord]:

    cmd = [
        SACCT,
        "-P",
        "-n",
        "-s",
        "COMPLETED",
        "--format",
        ",".join(ATTRIBUTES),
        "--starttime",
        starttime.isoformat(),
        "--endtime",
        endtime.isoformat(),
    ]

    if jobids:
        cmd += ["-j", ",".join(map(str, jobids))]
    elif uids:
        cmd += ["-u", ",".join(map(str, uids))]
    elif gids:
        cmd += ["-a", "-g", ",".join(map(str, gids))]

    try:
        output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as err:
        print("Error querying SlurmDB", err)
        sys.exit()

    # job_records has two levels: job_record[jobid][step]
    job_records = collections.defaultdict(dict)

    for line in output.splitlines():
        values = line.decode("utf-8").split("|")
        if "." in values[0]:
            jobid, step = values[0].split(".")
        else:
            jobid = values[0]
            step = ""
        uid = int(values[1]) if values[1] else None
        gid = int(values[2]) if values[2] else None
        start = to_datetime(values[3])
        end = to_datetime(values[4])
        state = values[5]
        alloccpus = int(values[6])
        reqmem = values[7]
        totalcpu = to_deltatime(values[8])
        avecpu = to_deltatime(values[9])
        avecpufreq = to_cpufreq(values[10])
        elapsed = to_deltatime(values[11])
        maxrss = to_memsize(values[12])
        exitcode, exitsignal = to_exitcode(values[13])
        nnodes = int(values[14])
        ntasks = int(values[15]) if values[15] else None
        nodelist = values[16].split(",")

        job_records[jobid][step] = JobRecord(
            jobid,
            step,
            uid,
            gid,
            start,
            end,
            state,
            alloccpus,
            reqmem,
            totalcpu,
            avecpu,
            avecpufreq,
            elapsed,
            maxrss,
            exitcode,
            exitsignal,
            nnodes,
            ntasks,
            nodelist,
        )

    return job_records


def to_exitcode(text) -> Tuple[Optional[int], Optional[int]]:

    if not text:
        exitcode = None
        exitsignal = None
    elif ":" in text:
        exitcode, exitsignal = [int(v) for v in text.split(":")]
    else:
        exitcode = int(text)
        exitsignal = None

    return exitcode, exitsignal


def to_datetime(text: str) -> Optional[datetime.datetime]:

    if text != "Unknown":
        return datetime.datetime.fromisoformat(text)


def to_deltatime(text: str) -> datetime.timedelta:

    if text == "":
        return

    if m := re.match(r"(\d+)-(\d{2}):(\d{2}):(\d{2})", text):
        days = int(m.group(1))
        hours = int(m.group(2))
        minutes = int(m.group(2))
        seconds = int(m.group(4))
        return datetime.timedelta(
            days=days, hours=hours, minutes=minutes, seconds=seconds
        )
    if m := re.match(r"(\d{2}):(\d{2}):(\d{2})", text):
        hours = int(m.group(1))
        minutes = int(m.group(2))
        seconds = int(m.group(3))
        return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    if m := re.match(r"(\d{2}):([\d\.]+)", text):
        minutes = int(m.group(1))
        seconds = float(m.group(2))
        return datetime.timedelta(minutes=minutes, seconds=seconds)


def to_memsize(text: str) -> Optional[int]:

    if not text:
        return
    if text[-1] == "G":
        return float(text[:-1]) * 1024 * 1024 * 1024
    if text[-1] == "M":
        return float(text[:-1]) * 1024 * 1024
    if text[-1] == "K":
        return float(text[:-1]) * 1024
    else:
        return int(text)


def to_cpufreq(text: str) -> Optional[int]:

    if not text:
        return
    if text[-1] == "K":
        return int(text[:-1])


def initials(uid) -> str:

    user = pwd.getpwuid(uid).pw_name
    return "".join([s[0].upper() for s in user.split(".")])
