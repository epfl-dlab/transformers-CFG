import pytest
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer
from dataclasses import dataclass


@dataclass
class CalFlowTestCase:
    name: str
    calflow: str


valid_cal_flow_sentences = [
    CalFlowTestCase("simplest_entry", "(Yield (toRecipient (CurrentUser)))"),
    CalFlowTestCase(
        "empty_struct_constraint",
        "(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (^(Event) EmptyStructConstraint))))",
    ),
    CalFlowTestCase(
        "find_reports", "(Yield (FindReports (toRecipient (CurrentUser))))"
    ),
    CalFlowTestCase(
        "long_entry",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.subject_? (?= "Work Shift")) (Event.start_? (?= (DateAtTimeWithDefaults (MDY 2L (April) (Year.apply 2019L)) (NumberAM 8L))))) (Event.end_? (DateTime.time_? (?= (NumberPM 4L))))))))',
    ),
    CalFlowTestCase(
        "find_manager",
        "(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.start_? (?= (DateAtTimeWithDefaults (Tomorrow) (HourMinuteMilitary 13L 0L)))) (Event.attendees_? (AttendeeListHasRecipient (FindManager (toRecipient (CurrentUser)))))))))",
    ),
    CalFlowTestCase(
        "attendees_constraint",
        "(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.start_? (DateTimeConstraint (Morning) (Tomorrow))) (Event.attendees_? (AttendeeListHasPeople (FindTeamOf (toRecipient (CurrentUser)))))))))",
    ),
    CalFlowTestCase(
        "next_dow_1",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.start_? (?= (Monday))) (Event.attendees_? (AttendeeListHasRecipient (Execute (refer (extensionConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "john")))))))))))',
    ),
    CalFlowTestCase(
        "next_dow",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.start_? (DateTime.date_? (?= (NextDOW (Monday))))) (Event.attendees_? (& (AttendeeListHasRecipient (Execute (refer (extensionConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "john")))))) (AttendeeListHasRecipient (FindManager (Execute (refer (extensionConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "john")))))))))))))',
    ),
    CalFlowTestCase(
        "event_duration",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.subject_? (?= "dinner prep")) (Event.start_? (DateTimeConstraint (Evening) (NextDOW (Wednesday))))) (Event.duration_? (?= (toMinutes 15)))))))',
    ),
    CalFlowTestCase(
        "holiday",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.subject_? (?= "party")) (Event.start_? (?= (DateAtTimeWithDefaults (NextHolidayFromToday (Holiday.ValentinesDay)) (NumberPM 7L))))) (Event.duration_? (?= (toHours 3)))))))',
    ),
    CalFlowTestCase(
        "location",
        "(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.location_? (roomRequest)) (Event.attendees_? (AttendeeListHasPeople (FindTeamOf (toRecipient (CurrentUser)))))))))",
    ),
    CalFlowTestCase(
        "weather",
        '(Yield (WeatherAggregate (WeatherQuantifier.Summarize) (temperature) (WeatherQueryApi (AtPlace (FindPlace (LocationKeyphrase.apply "DC Chinatown"))) (DateTime.date_? (?= (NextDOW (Saturday)))))))',
    ),
    CalFlowTestCase(
        "newxt_dow",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.subject_? (?= "lunch")) (Event.start_? (DateTime.date_? (?= (NextDOW (Saturday)))))) (Event.location_? (?= (LocationKeyphrase.apply "DC Chinatown")))))))',
    ),
    CalFlowTestCase(
        "do",
        '(do (Yield (WeatherAggregate (WeatherQuantifier.Summarize) (temperature) (WeatherQueryApi (AtPlace (FindPlace (LocationKeyphrase.apply "DC Chinatown"))) (DateTime.date_? (?= (NextDOW (Saturday))))))) (Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.subject_? (?= "lunch")) (Event.start_? (DateTime.date_? (?= (NextDOW (Saturday)))))) (Event.location_? (?= (LocationKeyphrase.apply "DC Chinatown"))))))))',
    ),
    CalFlowTestCase(
        "night",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.subject_? (?= "meal")) (Event.start_? (DateTimeConstraint (Night) (Today)))) (Event.attendees_? (& (AttendeeListHasPeople (FindTeamOf (toRecipient (CurrentUser)))) (AttendeeListExcludesRecipient (?= (Execute (refer (extensionConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "Abby")))))))))))))',
    ),
    CalFlowTestCase(
        "season",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.subject_? (?= "swimming party")) (Event.start_? (DateTime.date_? (SeasonSummer)))))))',
    ),
    CalFlowTestCase(
        "a_couple",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.subject_? (?= "lunch meeting")) (Event.start_? (DateTime.date_? (ThisWeek)))) (Event.duration_? (?= (toHours (longToNum (Acouple)))))))))',
    ),
    CalFlowTestCase(
        "room_request",
        "(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.start_? (DateTime.date_? (?= (NextDOW (Friday))))) (Event.location_? (roomRequest))))))",
    ),
    CalFlowTestCase(
        "multiple_attendees",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.subject_? (?= "finance meeting")) (Event.start_? (DateTime.date_? (?= (NextDOW (Tuesday)))))) (Event.attendees_? (& (& (& (AttendeeListHasRecipient (Execute (refer (extensionConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "Pam")))))) (AttendeeListHasRecipient (Execute (refer (extensionConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "Jim"))))))) (AttendeeListHasRecipient (Execute (refer (extensionConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "Michael"))))))) (AttendeeListHasRecipient (Execute (refer (extensionConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "Phyllis"))))))))))))',
    ),
    CalFlowTestCase(
        "next_month",
        "(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (EventAllDayStartingDateForPeriod (^(Event) EmptyStructConstraint) (MD 10L (NextMonth)) (toDays 3)))))",
    ),
    CalFlowTestCase(
        "status_busy",
        "(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.start_? (?= (Now))) (Event.end_? (?= (DateAtTimeWithDefaults (Tomorrow) (NumberPM 2L))))) (Event.showAs_? (?= (ShowAsStatus.Busy)))))))",
    ),
    CalFlowTestCase(
        "is_all_day",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.subject_? (?= "visit grandma")) (Event.location_? (?= (LocationKeyphrase.apply "Portugal")))) (Event.isAllDay_? (?= true))))))',
    ),
    CalFlowTestCase(
        "date_and_constraint",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (EventAllDayForDateRange (Event.subject_? (?= "pre super bowl party")) (DateAndConstraint (MD 12L (October)) (nextMonthDay (MD 12L (October)) (October) 17L))))))',
    ),
    CalFlowTestCase(
        "negate",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.subject_? (?= "girl\'s night")) (Event.start_? (DateTime.date_? (Date.dayOfWeek_? (negate (Weekend)))))))))',
    ),
    CalFlowTestCase(
        "hour_minute_pm",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.subject_? (?= "going sledding")) (Event.start_? (?= (DateAtTimeWithDefaults (Tomorrow) (HourMinutePm 6L 30L))))) (Event.end_? (?= (TimeAfterDateTime (DateAtTimeWithDefaults (Tomorrow) (HourMinutePm 6L 30L)) (HourMinutePm 7L 45L))))))))',
    ),
    CalFlowTestCase(
        "query_event_response",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.start_? (?= (Event.end (singleton (QueryEventResponse.results (FindEventWrapperWithDefaults (Event.subject_? (?~= "conference")))))))) (Event.duration_? (?= (toMinutes 30)))) (Event.showAs_? (?= (ShowAsStatus.Busy)))))))',
    ),
    CalFlowTestCase(
        "full_month",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.subject_? (?= "Baby Shower")) (Event.start_? (DateTime.date_? (FullMonthofMonth (September) (Year.apply 2023L))))))))',
    ),
    CalFlowTestCase(
        "event_on_day_after_time",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (EventOnDateAfterTime (NextDOW (Monday)) (Event.subject_? (?= "basketball game")) (DateTime.time (Event.end (singleton (QueryEventResponse.results (FindEventWrapperWithDefaults (Event.subject_? (?= "meeting")))))))))))',
    ),
    CalFlowTestCase(
        "adjust_by_duration",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.subject_? (?= "Dermatologist appointment")) (Event.start_? (?= (adjustByPeriodDuration (Event.end (singleton (QueryEventResponse.results (FindEventWrapperWithDefaults (EventOnDate (Today) (Event.subject_? (?~= "meeting"))))))) (PeriodDuration.apply :duration (toHours 1)))))))))',
    ),
    CalFlowTestCase(
        "plus_sign",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.subject_? (?= "tell Augie where you\'re going")) (Event.start_? (?= (PeriodDurationBeforeDateTime (Event.start (singleton (QueryEventResponse.results (FindEventWrapperWithDefaults (Event.attendees_? (AttendeeListHasRecipientConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "Adrielle")))))))) (PeriodDuration.apply :duration (toHours (+ 1 0.5))))))))))',
    ),
    CalFlowTestCase(
        "on_date_after_time",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.subject_? (?= "training time")) (Event.start_? (OnDateAfterTime (Tomorrow) (Lunch)))))))',
    ),
    CalFlowTestCase(
        "new_day_of_week",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (EventAllDayForDateRange (Event.location_? (?= (LocationKeyphrase.apply "London"))) (DateAndConstraint (DowOfWeekNew (Monday) (NextWeekList)) (nextDayOfWeek (DowOfWeekNew (Monday) (NextWeekList)) (Sunday)))))))',
    ),
    CalFlowTestCase(
        "next_day_of_month",
        "(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (EventAllDayStartingDateForPeriod (^(Event) EmptyStructConstraint) (nextDayOfMonth (Today) 4L) (toDays 3)))))",
    ),
    CalFlowTestCase(
        "hour_minute_am",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.subject_? (?= "anniversary")) (Event.start_? (?= (DateAtTimeWithDefaults (MD 31L (December)) (HourMinuteAm 11L 5L))))))))',
    ),
    CalFlowTestCase(
        "previous",
        '(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.subject_? (?= "haircut")) (Event.start_? (DateTime.date_? (?= (previousMonthDay (Today) (Date.month (PeriodBeforeDate (Today) (toMonths 1))) 12L))))))))',
    ),
]

valid_cal_flow_prefixes = [
    CalFlowTestCase("empty_string", ""),
    CalFlowTestCase(
        "unbalanced_paranthesis", "(Yield (FindReports (toRecipient (CurrentUser)))"
    ),
]

invalid_cal_flow_sentences = [
    CalFlowTestCase("empty_paranthesis", "()"),
    CalFlowTestCase("no_prefix_command", "(FindReports (toRecipient (CurrentUser)))"),
    CalFlowTestCase(
        "fake_day",
        "(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (Event.start_? (DateTime.date_? (?< (NextDOW (Pupday))))) (Event.attendees_? (AttendeeListHasRecipient (FindManager (toRecipient (CurrentUser)))))))))",
    ),
    CalFlowTestCase(
        "no_space_between_operators", "(Yield(FindReports(toRecipient(CurrentUser))))"
    ),
]


@pytest.fixture(scope="module")
def recognizer():
    with open(f"examples/grammars/calflow.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)
    start_rule_id = parsed_grammar.symbol_table["root"]
    recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)
    print("SetUp successful!", flush=True)
    return recognizer


def test_valid_sentence(recognizer):
    for cal_flow_test_case in valid_cal_flow_sentences:
        assert (
            recognizer._accept_string(cal_flow_test_case.calflow) == True
        ), f"Failed on {cal_flow_test_case.name}, {cal_flow_test_case.calflow}"

    for cal_flow_test_case in valid_cal_flow_prefixes + invalid_cal_flow_sentences:
        assert (
            recognizer._accept_string(cal_flow_test_case.calflow) == False
        ), f"Failed on {cal_flow_test_case.name}, {cal_flow_test_case.calflow}"


def test_valid_prefixes(recognizer):
    for cal_flow_test_case in valid_cal_flow_sentences + valid_cal_flow_prefixes:
        assert (
            recognizer._accept_prefix(cal_flow_test_case.calflow) == True
        ), f"Failed on {cal_flow_test_case.name}, {cal_flow_test_case.calflow}"

    for cal_flow_test_case in invalid_cal_flow_sentences:
        assert (
            recognizer._accept_prefix(cal_flow_test_case.calflow) == False
        ), f"Failed on {cal_flow_test_case.name}, {cal_flow_test_case.calflow}"
