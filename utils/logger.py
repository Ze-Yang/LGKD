import logging
import os
import sys
from termcolor import colored


class Logger:

    def __init__(self, logdir, rank, type='tensorboardX', debug=False, filename=None, summary=True, step=None):
        self.writer = None
        self.type = type
        self.rank = rank
        self.step = step

        self.summary = summary
        if summary:
            if type == 'tensorboardX':
                import tensorboardX
                self.writer = tensorboardX.SummaryWriter(logdir)
            else:
                raise NotImplementedError
        else:
            self.type = 'None'

        self.debug_flag = debug
        self.logger = self.setup_logger(output=logdir, distributed_rank=rank)

        if rank == 0:
            self.logger.info(f"[!] starting logging at directory {logdir}")
            if self.debug_flag:
                self.logger.info(f"[!] Entering DEBUG mode")

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.info("Closing the Writer.")

    def add_scalar(self, tag, scalar_value, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            self.writer.add_scalar(tag, scalar_value, step)

    def add_image(self, tag, image, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            self.writer.add_image(tag, image, step)

    def add_figure(self, tag, image, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            self.writer.add_figure(tag, image, step)

    def add_table(self, tag, tbl, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            tbl_str = "<table width=\"100%\"> "
            tbl_str += "<tr> \
                     <th>Term</th> \
                     <th>Value</th> \
                     </tr>"
            for k, v in tbl.items():
                tbl_str += "<tr> \
                           <td>%s</td> \
                           <td>%s</td> \
                           </tr>" % (k, v)

            tbl_str += "</table>"
            self.writer.add_text(tag, tbl_str, step)

    def info(self, msg):
        if self.rank == 0:
            self.logger.info(msg)

    def debug(self, msg):
        if self.rank == 0 and self.debug_flag:
            self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

    def _transform_tag(self, tag):
        tag = tag + f"/{self.step}" if self.step is not None else tag
        return tag

    def add_results(self, results):
        if self.type == 'tensorboardX':
            tag = self._transform_tag("Results")
            text = "<table width=\"100%\">"
            for k, res in results.items():
                text += f"<tr><td>{k}</td>" + " ".join([str(f'<td>{x}</td>') for x in res.values()]) + "</tr>"
            text += "</table>"
            self.writer.add_text(tag, text)

    def setup_logger(self, output=None, distributed_rank=0, color=True, name="PLOP", abbrev_name=None):
        """
        Args:
            output (str): a file name or a directory to save log. If None, will not save log file.
                If ends with ".txt" or ".log", assumed to be a file name.
                Otherwise, logs will be saved to `output/log.txt`.
            name (str): the root module name of this logger
            abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
                Set to "" to not log the root module in logs.
                By default, will abbreviate "detectron2" to "d2" and leave other
                modules unchanged.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if abbrev_name is None:
            abbrev_name = "plop" if name == "PLOP" else name

        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
        )
        # stdout logging: master only
        if distributed_rank == 0:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.DEBUG)
            if color:
                formatter = _ColorfulFormatter(
                    colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                    datefmt="%m/%d %H:%M:%S",
                    root_name=name,
                    abbrev_name=str(abbrev_name),
                )
            else:
                formatter = plain_formatter
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        # file logging
        if output is not None and distributed_rank == 0:
            if output.endswith(".txt") or output.endswith(".log"):
                filename = output
            else:
                filename = os.path.join(output, "log.txt")
            if distributed_rank > 0:
                filename = filename + ".rank{}".format(distributed_rank)
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            fh = logging.StreamHandler(_cached_log_stream(filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)

        return logger


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
def _cached_log_stream(filename):
    return open(filename, "a")
